#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local A/B spike for model-aware delegation in the Nemo OO coding agent."""

from __future__ import annotations

import argparse
import asyncio
import importlib.machinery
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from dotenv import dotenv_values
from nemo_oo_agents import Agent, PredictStrategy, hidden, strategy
from nemo_oo_agents.unifiedllm import CompletionClient, FakeLLMClient, UnifiedLLM
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    REPO_ROOT
    / "src"
    / "platform"
    / "agents"
    / "glamr"
    / "openshell"
    / "runners"
    / "nemo-oo"
    / "nemo-oo-coding-runner"
)
DEFAULT_API_BASE = "https://integrate.api.nvidia.com/v1"
MODEL_AWARE_GUIDANCE = """
A model-aware planning prepass has already classified work as `orchestrator` or
`worker`. The plan and any read-only worker findings are available in context.
Use those findings as advice, check them against the repository, and retain
responsibility for exploration, edits, integration, final judgment, and
verification.
"""
PROMPT_ONLY_GUIDANCE = """
You are the integrating engineer and have a cheaper, read-only coding analyst
available through self.delegate_bounded(task, context).

When creating the todo plan, classify work by execution role:
- Delegate crisp, bounded analysis when all necessary evidence can be supplied
  in the task and context, such as enumerating edge cases or critiquing a test
  matrix.
- Retain repository exploration, architectural decisions, edits, integration,
  final judgment, and verification.
- Prefer delegation when a suitable bounded subtask exists, but do not delegate
  work that requires repository access or authoritative verification.

Treat delegated output as advice and check it before acting. Select semantic
roles only; do not name or select models.
"""


def _load_runner_module() -> Any:
    loader = importlib.machinery.SourceFileLoader(
        "model_aware_demo_runner",
        str(RUNNER_PATH),
    )
    spec = importlib.util.spec_from_loader("model_aware_demo_runner", loader)
    if spec is None:
        raise RuntimeError(f"Could not create import spec for {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    runner_dir = str(RUNNER_PATH.parent)
    sys.path.insert(0, runner_dir)
    try:
        loader.exec_module(module)
    finally:
        sys.path.remove(runner_dir)
    return module


runner = _load_runner_module()
BenchAgent = runner.BenchAgent


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def _usage_int(usage: object, name: str) -> int:
    if isinstance(usage, dict):
        value = usage.get(name, 0)
    else:
        value = getattr(usage, name, 0)
    return int(value or 0)


class MeteredLLM(UnifiedLLM):
    """Transparent LLM wrapper that records provider-reported token usage."""

    def __init__(
        self,
        inner: UnifiedLLM,
        *,
        label: str,
        fallback_context_window: int,
    ) -> None:
        super().__init__(inner.model)
        self.inner = inner
        self.label = label
        self.fallback_context_window = fallback_context_window
        self.usage = Usage()

    @property
    def context_window(self) -> int | None:
        return self.inner.context_window or self.fallback_context_window

    def count_tokens(self, text: str) -> int:
        return self.inner.count_tokens(text)

    def count_tokens_raw(self, text: str) -> int:
        return self.inner.count_tokens_raw(text)

    def supports_vision(self) -> bool:
        return self.inner.supports_vision()

    def get_model_info(self) -> Any:
        return self.inner.get_model_info()

    def _record(self, response: Any) -> Any:
        usage = getattr(response, "usage", None)
        self.usage.calls += 1
        if usage is not None:
            self.usage.prompt_tokens += _usage_int(usage, "prompt_tokens")
            self.usage.completion_tokens += _usage_int(usage, "completion_tokens")
            self.usage.total_tokens += _usage_int(usage, "total_tokens")
        return response

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        output_model: type[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        response = self.inner.call(
            messages=messages,
            tools=tools,
            output_model=output_model,
            **kwargs,
        )
        return self._record(response)

    async def acall(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        output_model: type[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        response = await self.inner.acall(
            messages=messages,
            tools=tools,
            output_model=output_model,
            **kwargs,
        )
        return self._record(response)


class BoundedCodingAnalyst(Agent, llm=FakeLLMClient()):
    """A read-only analyst with no repository or execution tools."""

    @strategy(PredictStrategy())
    async def analyze(self, task: str, context: str = "") -> str:
        """Complete one bounded software-analysis task.

        Work only from the supplied task and context. Do not claim to inspect
        files, run commands, or verify facts that are not present. Return a
        concise recommendation with supporting evidence, edge cases, and any
        uncertainty the integrating engineer should check.
        """
        ...


class PlannedStep(BaseModel):
    """One model-aware coding-plan step."""

    task: str
    target: Literal["orchestrator", "worker"]
    reason: str


class ModelAwarePlan(BaseModel):
    """A plan that separates integration work from bounded analysis."""

    steps: list[PlannedStep]


class PromptOnlyModelAwareBenchAgent(BenchAgent):
    """BenchAgent whose frontier loop decides whether to delegate."""

    analyst: Annotated[BoundedCodingAnalyst, hidden]
    delegation_log: Annotated[list[dict[str, str]], hidden]

    def __init__(
        self,
        llm: UnifiedLLM,
        *,
        worker_llm: UnifiedLLM,
    ) -> None:
        super().__init__(llm=llm)
        self.analyst = BoundedCodingAnalyst(llm=worker_llm)
        self.delegation_log = []
        self.context_manager.set_static(
            "model_aware_delegation",
            PROMPT_ONLY_GUIDANCE,
        )

    def _seed_todos(self) -> None:
        self.todo.add("Plan the task and decide which steps to delegate versus retain")

    async def delegate_bounded(self, task: str, context: str = "") -> str:
        """Ask the cheaper read-only analyst one crisp question.

        Supply all evidence needed in context. Good uses include reviewing a
        short excerpt, enumerating edge cases, or critiquing a proposed test
        matrix. Do not use this for repository exploration, edits, final
        decisions, or verification.
        """
        response = await self.analyst.analyze(task, context)
        self.delegation_log.append(
            {
                "task": task,
                "context": context,
                "response": response,
            }
        )
        return response


class ModelAwareBenchAgent(BenchAgent):
    """BenchAgent variant with one cheaper read-only delegation capability."""

    analyst: Annotated[BoundedCodingAnalyst, hidden]
    delegation_log: Annotated[list[dict[str, str]], hidden]
    model_aware_plan: Annotated[ModelAwarePlan | None, hidden]

    def __init__(
        self,
        llm: UnifiedLLM,
        *,
        worker_llm: UnifiedLLM,
    ) -> None:
        super().__init__(llm=llm)
        self.analyst = BoundedCodingAnalyst(llm=worker_llm)
        self.delegation_log = []
        self.model_aware_plan = None
        self.context_manager.set_static(
            "model_aware_delegation",
            MODEL_AWARE_GUIDANCE,
        )

    def _seed_todos(self) -> None:
        self.todo.add("Review the model-aware plan and delegated findings")

    @strategy(PredictStrategy())
    async def create_model_aware_plan(self, task: str) -> ModelAwarePlan:
        """Plan a coding task by semantic execution role.

        Return a short ordered plan. Tag bounded, read-only analysis as
        `worker` when it has a crisp deliverable and can be completed from the
        task text alone, such as enumerating edge cases or critiquing a test
        matrix. Tag repository exploration, architectural choices, edits,
        integration, and verification as `orchestrator`.

        A non-trivial coding task with explicit edge cases or test requirements
        should normally include at least one `worker` step. Do not name or pick
        models.
        """
        ...

    async def _run_evaluation(
        self,
        task_input: dict[str, object],
    ) -> dict[str, object]:
        problem_statement = runner._problem_statement(task_input)
        self.model_aware_plan = await self.create_model_aware_plan(problem_statement)
        worker_steps = [step for step in self.model_aware_plan.steps if step.target == "worker"][:3]
        if worker_steps:
            await asyncio.gather(
                *(
                    self.delegate_bounded(
                        step.task,
                        context=problem_statement,
                    )
                    for step in worker_steps
                )
            )

        self.context_manager.set_static(
            "model_aware_plan",
            self.model_aware_plan.model_dump_json(indent=2),
        )
        self.context_manager.set_static(
            "delegated_findings",
            json.dumps(self.delegation_log, indent=2),
        )
        return await super()._run_evaluation(task_input)

    async def delegate_bounded(self, task: str, context: str = "") -> str:
        """Ask the cheaper read-only analyst one crisp question.

        Supply all evidence needed in context. Good uses include reviewing a
        short excerpt, enumerating edge cases, or critiquing a proposed test
        matrix. Do not use this for repository exploration, edits, final
        decisions, or verification.
        """
        response = await self.analyst.analyze(task, context)
        self.delegation_log.append(
            {
                "task": task,
                "context": context,
                "response": response,
            }
        )
        return response


def _client(
    *,
    model: str,
    api_base: str,
    api_key: str,
    label: str,
    context_window: int,
) -> MeteredLLM:
    inner = CompletionClient(
        model=f"openai/{model}",
        api_base=api_base,
        api_key=api_key if api_base.startswith("https://") else "unused",
        drop_params=True,
    )
    return MeteredLLM(
        inner,
        label=label,
        fallback_context_window=context_window,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("baseline", "model-aware"), required=True)
    parser.add_argument(
        "--mode",
        choices=("prompt-only", "structured"),
        default="prompt-only",
    )
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="nvidia/nemotron-3-super-120b-a12b",
    )
    parser.add_argument(
        "--worker-model",
        default="nvidia/nemotron-3-nano-30b-a3b",
    )
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--worker-api-base")
    parser.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    parser.add_argument("--context-window", type=int, default=32768)
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> dict[str, object]:
    env = dotenv_values(REPO_ROOT / ".env")
    api_key = str(os.getenv(args.api_key_env) or env.get(args.api_key_env) or "")
    if args.api_base.startswith("https://") and not api_key:
        raise RuntimeError(f"{args.api_key_env} is not configured in the environment or repo .env")

    primary = _client(
        model=args.model,
        api_base=args.api_base,
        api_key=api_key,
        label="orchestrator",
        context_window=args.context_window,
    )
    worker: MeteredLLM | None = None
    if args.variant == "model-aware":
        worker = _client(
            model=args.worker_model,
            api_base=args.worker_api_base or args.api_base,
            api_key=api_key,
            label="bounded-worker",
            context_window=args.context_window,
        )
        if args.mode == "structured":
            agent = ModelAwareBenchAgent(llm=primary, worker_llm=worker)
        else:
            agent = PromptOnlyModelAwareBenchAgent(
                llm=primary,
                worker_llm=worker,
            )
    else:
        agent = BenchAgent(llm=primary)

    task = args.task_file.read_text()
    started_at = time.perf_counter()
    evaluation = await agent._run_evaluation(
        {
            "problem_statement": task,
            "working_dir": str(args.workdir.resolve()),
        }
    )
    elapsed_seconds = time.perf_counter() - started_at
    delegations = (
        agent.delegation_log
        if isinstance(
            agent,
            (ModelAwareBenchAgent, PromptOnlyModelAwareBenchAgent),
        )
        else []
    )
    plan = (
        agent.model_aware_plan.model_dump()
        if isinstance(agent, ModelAwareBenchAgent) and agent.model_aware_plan is not None
        else None
    )
    return {
        "variant": args.variant,
        "mode": args.mode if args.variant == "model-aware" else None,
        "model": args.model,
        "worker_model": args.worker_model if worker is not None else None,
        "evaluation": evaluation,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "plan": plan,
        "orchestrator_usage": asdict(primary.usage),
        "worker_usage": asdict(worker.usage) if worker is not None else None,
        "delegations": delegations,
    }


def main() -> None:
    args = _parse_args()
    report = asyncio.run(_run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
