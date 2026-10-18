"""
Rate limiting for every way into HawkEye.

Two kinds of limiter, chosen per surface:

* Database-backed (questions). Each question is already a row in `queries`,
  inserted the moment it starts, so the quota check just counts the asker's
  recent rows. Nothing extra to store, survives restarts and deploys, and would
  keep holding if the app ever ran on more than one machine. Questions are the
  expensive surface -- several Gemini calls each -- so they get the durable one.
* In-memory sliding windows (login attempts per IP, feedback clicks, admin tab
  actions, the one-question-at-a-time guard). Cheap, high-frequency checks where
  a database round trip per click would itself be something to abuse. They reset
  on restart and assume a single app machine, which is what fly.toml runs.

Every limit can be overridden with an environment variable (the RATE_LIMIT_*
names below); 0 disables a window.
"""
import math
import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass

from implementation.db import db_pool


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


# ---------------------------------------------------------------------------
# The refusal
# ---------------------------------------------------------------------------

class RateLimited(Exception):
    """
    A refused request. `scope` names the limit for the audit log; user_message()
    is what the technician sees -- a plain sentence with a time to wait, never a
    stack trace or a generic error.
    """

    def __init__(self, scope: str, retry_after: float, limit: int | None = None,
                 period: str | None = None, message: str | None = None):
        self.scope = scope
        self.retry_after = max(0.0, float(retry_after))
        self.limit = limit
        self.period = period
        self._message = message
        super().__init__(self.user_message())

    def user_message(self) -> str:
        if self._message:
            return self._message
        what = f"the limit of {self.limit} questions per {self.period}" if self.limit else "a usage limit"
        return f"You've reached {what}. Try again in {format_wait(self.retry_after)}."

    def audit_details(self) -> dict:
        return {"scope": self.scope, "limit": self.limit, "period": self.period,
                "retry_after_seconds": math.ceil(self.retry_after)}


def format_wait(seconds: float) -> str:
    seconds = max(1, math.ceil(seconds))
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    minutes = math.ceil(seconds / 60)
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours = math.ceil(minutes / 60)
    return f"{hours} hour{'s' if hours != 1 else ''}"


# ---------------------------------------------------------------------------
# In-memory sliding window
# ---------------------------------------------------------------------------

class SlidingWindowLimiter:
    """
    At most `limit` hits per key in any rolling `seconds`-long window. Thread-safe
    (Gradio runs handlers on a thread pool). Idle keys are swept periodically so
    a stream of one-off IPs can't grow the table forever.
    """

    SWEEP_EVERY = 1000

    def __init__(self, limit: int, seconds: float, clock=time.monotonic):
        self.limit = limit
        self.seconds = seconds
        self._clock = clock
        self._hits: dict[object, deque] = {}
        self._lock = threading.Lock()
        self._calls = 0

    def hit(self, key) -> float | None:
        """Record one hit. Returns None if allowed, else seconds until it would be."""
        if self.limit <= 0:
            return None
        now = self._clock()
        with self._lock:
            self._calls += 1
            if self._calls % self.SWEEP_EVERY == 0:
                self._sweep(now)
            window = self._hits.setdefault(key, deque())
            while window and window[0] <= now - self.seconds:
                window.popleft()
            if len(window) >= self.limit:
                return window[0] + self.seconds - now
            window.append(now)
            return None

    def _sweep(self, now: float) -> None:
        for key in [k for k, w in self._hits.items() if not w or w[-1] <= now - self.seconds]:
            del self._hits[key]


# Sign-in attempts per client IP, enforced in app.py's middleware before Gradio
# even sees the request. The per-username lockout lives in accounts.py.
login_ip_limiter = SlidingWindowLimiter(
    _env_int("RATE_LIMIT_LOGIN_PER_IP", 10), _env_int("RATE_LIMIT_LOGIN_IP_WINDOW_SECONDS", 300)
)
# Thumbs up/down clicks per user.
feedback_limiter = SlidingWindowLimiter(_env_int("RATE_LIMIT_FEEDBACK_PER_MINUTE", 30), 60)
# Any admin-tab action (loading Refresh History, the map, Accounts, Activity).
admin_action_limiter = SlidingWindowLimiter(_env_int("RATE_LIMIT_ADMIN_ACTIONS_PER_MINUTE", 30), 60)
# Accounts created per admin, across single and bulk creation.
account_creation_limiter = SlidingWindowLimiter(_env_int("RATE_LIMIT_ACCOUNTS_PER_HOUR", 50), 3600)
# Caps how many rate_limited / access_denied rows one user or IP can add to
# audit_events, so a spammer being refused can't flood the audit log instead.
audit_noise_limiter = SlidingWindowLimiter(10, 60)


# ---------------------------------------------------------------------------
# One question at a time per user
# ---------------------------------------------------------------------------

class InFlightGuard:
    """
    Refuses a second question from the same user while their first is still
    being answered (two tabs, double-clicks, scripts). A lease older than
    `stale_after` is treated as abandoned -- if a stream is never closed, its
    owner isn't locked out forever.
    """

    def __init__(self, stale_after: float = 180, clock=time.monotonic):
        self.stale_after = stale_after
        self._clock = clock
        self._held: dict[object, float] = {}
        self._lock = threading.Lock()

    def acquire(self, key) -> None:
        now = self._clock()
        with self._lock:
            started = self._held.get(key)
            if started is not None and now - started < self.stale_after:
                raise RateLimited(
                    "concurrent", retry_after=0,
                    message="HawkEye is still answering your previous question. "
                            "Wait for it to finish, then ask again.",
                )
            self._held[key] = now

    def release(self, key) -> None:
        with self._lock:
            self._held.pop(key, None)

    @contextmanager
    def hold(self, key):
        self.acquire(key)
        try:
            yield
        finally:
            self.release(key)


question_guard = InFlightGuard()


# ---------------------------------------------------------------------------
# Question quotas, counted from the queries table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuestionPolicy:
    per_minute: int
    per_hour: int
    per_day: int


QUESTION_POLICIES = {
    "worker": QuestionPolicy(
        per_minute=_env_int("RATE_LIMIT_WORKER_PER_MINUTE", 5),
        per_hour=_env_int("RATE_LIMIT_WORKER_PER_HOUR", 60),
        per_day=_env_int("RATE_LIMIT_WORKER_PER_DAY", 300),
    ),
    "admin": QuestionPolicy(
        per_minute=_env_int("RATE_LIMIT_ADMIN_PER_MINUTE", 10),
        per_hour=_env_int("RATE_LIMIT_ADMIN_PER_HOUR", 0),
        per_day=_env_int("RATE_LIMIT_ADMIN_PER_DAY", 0),
    ),
}

# Across everyone, rolling 24 hours: a ceiling on Gemini spend however many
# accounts are involved. Hitting it means something is wrong, not that the help
# desk is busy -- normal use is far below this.
GLOBAL_DAILY_QUESTIONS = _env_int("RATE_LIMIT_GLOBAL_PER_DAY", 3000)

WINDOW_SECONDS = {"minute": 60, "hour": 3600, "day": 86400}


@dataclass(frozen=True)
class QuotaUsage:
    """Questions in each rolling window, and the age in seconds of the oldest one in it."""
    minute: int
    hour: int
    day: int
    oldest_minute_age: float | None
    oldest_hour_age: float | None
    oldest_day_age: float | None
    global_day: int


def evaluate_quota(usage: QuotaUsage, policy: QuestionPolicy, global_limit: int) -> RateLimited | None:
    """Pure decision: the first exceeded limit, or None. Split out to be testable offline."""
    if global_limit > 0 and usage.global_day >= global_limit:
        return RateLimited(
            "global_day", retry_after=3600, limit=global_limit, period="day",
            message="HawkEye has hit its daily question limit for the whole help desk. "
                    "Try again later, or ask an admin if it's urgent.",
        )
    for period, limit, count, oldest_age in (
        ("minute", policy.per_minute, usage.minute, usage.oldest_minute_age),
        ("hour", policy.per_hour, usage.hour, usage.oldest_hour_age),
        ("day", policy.per_day, usage.day, usage.oldest_day_age),
    ):
        if limit > 0 and count >= limit:
            # The window frees a slot when its oldest question ages out of it.
            retry = WINDOW_SECONDS[period] - (oldest_age or 0)
            return RateLimited(f"user_{period}", retry_after=retry, limit=limit, period=period)
    return None


def fetch_quota_usage(user_id: int) -> QuotaUsage:
    """One round trip, served by the (user_id, created_at) and (created_at) indexes."""
    with db_pool.connection() as conn:
        row = conn.execute(
            """
            SELECT
                count(*) FILTER (WHERE created_at > now() - interval '1 minute'),
                count(*) FILTER (WHERE created_at > now() - interval '1 hour'),
                count(*),
                extract(epoch FROM now() - min(created_at) FILTER (WHERE created_at > now() - interval '1 minute')),
                extract(epoch FROM now() - min(created_at) FILTER (WHERE created_at > now() - interval '1 hour')),
                extract(epoch FROM now() - min(created_at)),
                (SELECT count(*) FROM queries WHERE created_at > now() - interval '1 day')
            FROM queries
            WHERE user_id = %s AND created_at > now() - interval '1 day'
            """,
            (user_id,),
        ).fetchone()
    as_float = lambda v: float(v) if v is not None else None  # noqa: E731
    return QuotaUsage(
        minute=row[0], hour=row[1], day=row[2],
        oldest_minute_age=as_float(row[3]), oldest_hour_age=as_float(row[4]),
        oldest_day_age=as_float(row[5]), global_day=row[6],
    )


def check_question_quota(user) -> None:
    """Raise RateLimited if `user` (an accounts.User) may not ask another question right now."""
    policy = QUESTION_POLICIES.get(user.role, QUESTION_POLICIES["worker"])
    refusal = evaluate_quota(fetch_quota_usage(user.id), policy, GLOBAL_DAILY_QUESTIONS)
    if refusal is not None:
        raise refusal


def check(limiter: SlidingWindowLimiter, key, scope: str, message: str) -> None:
    """Raise RateLimited with a fixed message if `limiter` refuses `key`."""
    retry = limiter.hit(key)
    if retry is not None:
        raise RateLimited(scope, retry_after=retry,
                          message=f"{message} Try again in {format_wait(retry)}.")
