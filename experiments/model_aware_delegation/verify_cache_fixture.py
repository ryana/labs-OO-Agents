#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent verifier for the async SWR cache experiment."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


class ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


async def verify_stale_refresh(cache_type: type) -> None:
    clock = ManualClock()
    cache = cache_type(fresh_ttl=10.0, stale_ttl=20.0, clock=clock)
    calls = 0

    async def initial_loader() -> str:
        nonlocal calls
        calls += 1
        return "old"

    assert await cache.get("key", initial_loader) == "old"
    clock.advance(10.0)
    started = asyncio.Event()
    release = asyncio.Event()

    async def refresh_loader() -> str:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return "new"

    assert await asyncio.wait_for(cache.get("key", refresh_loader), timeout=0.2) == "old"
    await asyncio.wait_for(started.wait(), timeout=1.0)
    assert await asyncio.wait_for(cache.get("key", refresh_loader), timeout=0.2) == "old"
    assert calls == 2
    release.set()
    await asyncio.wait_for(cache.wait_for_refreshes(), timeout=1.0)
    assert await cache.get("key", refresh_loader) == "new"
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1
    assert cache.stats.stale_hits == 2
    assert cache.stats.refreshes == 1
    assert cache.stats.refresh_failures == 0


async def verify_miss_coalescing(cache_type: type) -> None:
    cache = cache_type(fresh_ttl=10.0, stale_ttl=20.0)
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def loader() -> str:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return "value"

    tasks = [asyncio.create_task(cache.get("key", loader)) for _ in range(5)]
    await asyncio.wait_for(started.wait(), timeout=1.0)
    await asyncio.sleep(0)
    assert calls == 1
    release.set()
    assert await asyncio.gather(*tasks) == ["value"] * 5
    assert cache.stats.misses == 5


async def verify_per_key_parallelism(cache_type: type) -> None:
    cache = cache_type(fresh_ttl=10.0, stale_ttl=20.0)
    both_started = asyncio.Event()
    release = asyncio.Event()
    started: set[str] = set()

    def loader_for(key: str):
        async def loader() -> str:
            started.add(key)
            if started == {"a", "b"}:
                both_started.set()
            await release.wait()
            return key.upper()

        return loader

    task_a = asyncio.create_task(cache.get("a", loader_for("a")))
    task_b = asyncio.create_task(cache.get("b", loader_for("b")))
    await asyncio.wait_for(both_started.wait(), timeout=1.0)
    release.set()
    assert await asyncio.gather(task_a, task_b) == ["A", "B"]


async def verify_expired_waits(cache_type: type) -> None:
    clock = ManualClock()
    cache = cache_type(fresh_ttl=10.0, stale_ttl=20.0, clock=clock)

    async def initial_loader() -> str:
        return "old"

    assert await cache.get("key", initial_loader) == "old"
    clock.advance(30.0)
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def loader() -> str:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return "new"

    tasks = [asyncio.create_task(cache.get("key", loader)) for _ in range(2)]
    await asyncio.wait_for(started.wait(), timeout=1.0)
    await asyncio.sleep(0)
    assert not any(task.done() for task in tasks)
    assert calls == 1
    release.set()
    assert await asyncio.gather(*tasks) == ["new", "new"]
    assert cache.stats.misses == 3


async def verify_refresh_failure(cache_type: type) -> None:
    clock = ManualClock()
    cache = cache_type(fresh_ttl=1.0, stale_ttl=20.0, clock=clock)

    async def initial_loader() -> str:
        return "old"

    assert await cache.get("key", initial_loader) == "old"
    clock.advance(1.0)

    async def failing_loader() -> str:
        raise RuntimeError("refresh failed")

    assert await asyncio.wait_for(cache.get("key", failing_loader), timeout=0.2) == "old"
    await asyncio.wait_for(cache.wait_for_refreshes(), timeout=1.0)
    assert cache.stats.refresh_failures == 1

    async def recovery_loader() -> str:
        return "recovered"

    assert await asyncio.wait_for(cache.get("key", recovery_loader), timeout=0.2) == "old"
    await asyncio.wait_for(cache.wait_for_refreshes(), timeout=1.0)
    assert await cache.get("key", recovery_loader) == "recovered"


async def verify_failed_blocking_load(cache_type: type) -> None:
    cache = cache_type(fresh_ttl=10.0, stale_ttl=20.0)

    async def failing_loader() -> str:
        raise RuntimeError("load failed")

    try:
        await cache.get("key", failing_loader)
    except RuntimeError as exc:
        assert str(exc) == "load failed"
    else:
        raise AssertionError("blocking load failure should propagate")

    async def recovery_loader() -> str:
        return "recovered"

    assert await cache.get("key", recovery_loader) == "recovered"


async def verify_invalidation_race(cache_type: type) -> None:
    clock = ManualClock()
    cache = cache_type(fresh_ttl=1.0, stale_ttl=20.0, clock=clock)

    async def initial_loader() -> str:
        return "old"

    assert await cache.get("key", initial_loader) == "old"
    clock.advance(1.0)
    started = asyncio.Event()
    release = asyncio.Event()

    async def stale_refresh() -> str:
        started.set()
        await release.wait()
        return "resurrected"

    assert await asyncio.wait_for(cache.get("key", stale_refresh), timeout=0.2) == "old"
    await asyncio.wait_for(started.wait(), timeout=1.0)
    cache.invalidate("key")
    release.set()
    await asyncio.wait_for(cache.wait_for_refreshes(), timeout=1.0)

    async def new_loader() -> str:
        return "new"

    assert await cache.get("key", new_loader) == "new"


async def main_async(workdir: Path) -> None:
    sys.path.insert(0, str(workdir))
    try:
        from swr_cache import AsyncSWRCache

        await verify_stale_refresh(AsyncSWRCache)
        await verify_miss_coalescing(AsyncSWRCache)
        await verify_per_key_parallelism(AsyncSWRCache)
        await verify_expired_waits(AsyncSWRCache)
        await verify_refresh_failure(AsyncSWRCache)
        await verify_failed_blocking_load(AsyncSWRCache)
        await verify_invalidation_race(AsyncSWRCache)
    finally:
        sys.path.remove(str(workdir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", type=Path)
    args = parser.parse_args()
    asyncio.run(
        asyncio.wait_for(
            main_async(args.workdir.resolve()),
            timeout=10.0,
        )
    )
    print("independent async cache verifier passed")


if __name__ == "__main__":
    main()
