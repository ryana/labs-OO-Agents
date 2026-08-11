#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent verifier for the local Retry-After fixture."""

from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", type=Path)
    args = parser.parse_args()

    module_path = args.workdir / "retry_after.py"
    spec = importlib.util.spec_from_file_location("retry_after_under_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    now = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)  # noqa: UP017
    assert module.parse_retry_after("120", now=now) == 120.0
    assert (
        module.parse_retry_after(
            "Thu, 30 Jul 2026 12:02:00 GMT",
            now=now,
        )
        == 120.0
    )
    assert (
        module.parse_retry_after(
            "Thu, 30 Jul 2026 11:59:00 GMT",
            now=now,
        )
        == 0.0
    )

    invalid = ("-1", "1.5", "tomorrow")
    for value in invalid:
        try:
            module.parse_retry_after(value, now=now)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{value!r} should raise ValueError")

    try:
        module.parse_retry_after("1", now=now.replace(tzinfo=None))
    except ValueError:
        pass
    else:
        raise AssertionError("naive now should raise ValueError")

    print("independent verifier passed")


if __name__ == "__main__":
    main()
