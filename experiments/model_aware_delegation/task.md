Implement complete HTTP `Retry-After` parsing in `retry_after.py`.

Requirements:

- Preserve support for non-negative integer delta-seconds.
- Also support an IMF-fixdate such as `Thu, 30 Jul 2026 12:02:00 GMT`.
- Return `0.0` when an HTTP date is in the past.
- Reject negative delta-seconds, fractional delta-seconds, malformed values,
  and a naive `now` with `ValueError`.
- Do not add third-party dependencies.
- Add focused unit tests for the new behavior and run them before finishing.
