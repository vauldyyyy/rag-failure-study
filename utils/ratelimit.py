"""Client-side rate limiting.

The first real run died here, and the error was precise:

    quota_id: "EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier"
    quota_value: 100

Two things that were wrong and are fixed here:

1. **Batching does not save quota.** `embed_content` with a list of 100 texts is
   billed as 100 requests, not one. Batching only saves HTTP round trips. So the
   limiter counts *items*, not calls.

2. **Exponential backoff from 2s cannot clear a per-minute window.** The old ladder
   (2, 4, 8, 16s) retried five times inside the same exhausted minute and gave up
   after ~30 seconds. A per-minute quota needs a wait measured against the window,
   not against the attempt number.

The limiter is a sliding window: it records the timestamp of every request and
blocks until the oldest one falls out of the window. That is stricter than a token
bucket and cannot burst past the limit on the first call, which is exactly the
failure that happened.
"""

import threading
import time
from collections import deque

WINDOW_SECONDS = 60.0


class RateLimiter:
    def __init__(self, rpm: int, name: str = ""):
        self.rpm = max(1, int(rpm))
        self.name = name
        self._times = deque()
        self._lock = threading.Lock()
        self._announced = False

    def acquire(self, n: int = 1, verbose: bool = True):
        """Block until `n` more requests fit inside the window."""
        for _ in range(n):
            self._acquire_one(verbose)

    def _acquire_one(self, verbose):
        while True:
            with self._lock:
                now = time.monotonic()
                while self._times and now - self._times[0] >= WINDOW_SECONDS:
                    self._times.popleft()

                if len(self._times) < self.rpm:
                    self._times.append(now)
                    return

                wait = WINDOW_SECONDS - (now - self._times[0]) + 0.05

            if verbose and wait > 5 and not self._announced:
                print(f"  [{self.name}] at {self.rpm}/min limit, pacing "
                      f"({wait:.0f}s)...", end="\r")
                self._announced = True
            time.sleep(min(wait, WINDOW_SECONDS))
            self._announced = False

    def note_rejection(self):
        """Called after a 429: treat the window as fully spent.

        The server disagreed with our accounting, so trust the server. Filling the
        window forces the next acquire() to wait it out rather than immediately
        retrying into the same exhausted quota.
        """
        with self._lock:
            now = time.monotonic()
            self._times.clear()
            self._times.extend([now] * self.rpm)


def is_quota_error(e) -> bool:
    msg = str(e).lower()
    return ("429" in msg or "resource_exhausted" in msg or "resourceexhausted" in msg
            or "exceeded your current quota" in msg or "rate limit" in msg)


def is_daily_quota_error(e) -> bool:
    """A per-day cap will not clear by waiting a minute; the run must stop."""
    msg = str(e).lower()
    return is_quota_error(e) and ("perday" in msg.replace(" ", "")
                                  or "per day" in msg
                                  or "requestsperday" in msg.replace(" ", ""))


class DailyQuotaExhausted(RuntimeError):
    """Raised so a run can checkpoint and exit cleanly instead of thrashing."""


def quota_sleep(attempt: int) -> float:
    """Backoff for a per-minute quota rejection.

    Starts at a full window because that is what the quota actually needs, then
    grows slightly to ride out a busy shared project.
    """
    return min(180.0, WINDOW_SECONDS * (1 + 0.5 * attempt))
