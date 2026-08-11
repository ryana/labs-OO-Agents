# Model-aware coding-agent delegation spike

## Research question

Can a frontier coding agent use its knowledge of the plan to decide when to
delegate a bounded subtask to a cheaper model, while selecting only a semantic
role rather than a concrete model?

## Experiment design

`demo.py` wraps the existing GLAMR NeMo OO coding agent with one optional,
read-only `delegate_bounded(task, context)` capability. The orchestrator decides
whether to call it and what to ask. The harness binds that worker role to a
separately configured model and records provider-reported token usage.

This directory is the original local spike copied into the public fork so the
prompts, harness, verifiers, and raw reports are visible. The harness still
loads the GLAMR coding runner from the Agent Hub checkout; it is not a
standalone example for this repository.

## Files and how to run

- `demo.py`: A/B harness used for baseline and prompt-only model-aware runs.
- `task.md` and `verify_fixture.py`: small Retry-After parser task.
- `cache-task.md` and `verify_cache_fixture.py`: larger async
  stale-while-revalidate cache task.
- `reports/`: raw JSON output from the frontier-model runs.

From the original Agent Hub checkout, the runs used `uv run python` with
`--variant baseline` or `--variant model-aware`, separate clean fixture
directories, and an output JSON path. Run `uv run python demo.py --help` for the
full argument list. The NVIDIA endpoint key was supplied through an environment
variable; no credentials are included here.

## Key metrics and results

All async-cache variants passed their visible tests and the same independent
verifier. These are single runs, not a statistical benchmark.

| Variant | Orchestrator calls / tokens | Worker calls / tokens | Total tokens | Elapsed |
| --- | ---: | ---: | ---: | ---: |
| GPT-5.6 Sol only | 9 / 107,618 | — | 107,618 | 134.905 s |
| GPT-5.6 Sol + GPT-5.6 Luna | 8 / 99,558 | 1 / 1,901 | 101,459 | 122.842 s |
| GPT-5.6 Sol + DeepSeek V4 Flash | 9 / 136,544 | 1 / 1,862 | 138,406 | 202.724 s |

The Sol/Luna cache run displaced one frontier call and used 5.7% fewer total
tokens than Sol-only. The smaller Retry-After task moved in the other direction:
delegation increased total tokens from 50,419 to 63,405. The useful hypothesis
is therefore narrower: delegation pays when bounded worker output replaces
frontier work instead of merely adding another call.
