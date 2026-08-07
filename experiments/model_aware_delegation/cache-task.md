Implement complete stale-while-revalidate behavior in the `swr_cache` package.

Keep the existing public API:

- `AsyncSWRCache(fresh_ttl, stale_ttl, clock=...)`
- `await cache.get(key, loader)`
- `cache.invalidate(key)`
- `cache.stats`
- `await cache.wait_for_refreshes()`

Required behavior:

- A fresh entry returns immediately without calling its loader.
- A stale entry returns immediately and starts one background refresh for that
  key. Additional stale callers share that refresh and still receive the stale
  value immediately.
- An entry is stale once its age reaches `fresh_ttl`. It is fully expired once
  its age reaches `fresh_ttl + stale_ttl`.
- A missing or fully expired entry waits for a load. Concurrent callers for the
  same key must await one shared loader execution.
- Work is coalesced per key, not globally; different keys may load concurrently.
- A failed background refresh increments `refresh_failures`, does not escape as
  an unhandled task exception, and preserves the stale entry.
- A failed blocking load propagates to its callers and does not populate the
  cache.
- `invalidate(key)` removes the entry and prevents any already-running load or
  refresh from restoring that key. Existing callers may still receive the
  completed loader result.
- `wait_for_refreshes()` waits for all loads or refreshes that were already
  running when it was called. Background failures must not escape from it.
- Keep `CacheStats` immutable. Count:
  - `hits`: each fresh cached return
  - `stale_hits`: each stale cached return
  - `misses`: each caller that has no usable cached value and must await a load,
    including callers joining an existing load
  - `refreshes`: each background refresh started for a stale entry
  - `refresh_failures`: each failed background refresh
- Do not add third-party dependencies.
- Add focused deterministic tests, including concurrency, refresh failure, and
  invalidation races. Run the tests before finishing.
