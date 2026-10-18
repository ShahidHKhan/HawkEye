"""
Offline tests for implementation/rate_limit.py: the sliding window, the
one-question-at-a-time guard, and the quota decision. A fake clock drives time;
nothing touches the database.

    python tests/test_rate_limit.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://placeholder/unused")
os.environ.setdefault("GOOGLE_API_KEY", "placeholder-unused")

from implementation.rate_limit import (  # noqa: E402
    InFlightGuard, QuestionPolicy, QuotaUsage, RateLimited, SlidingWindowLimiter,
    evaluate_quota, format_wait,
)
from _runner import run_tests  # noqa: E402


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


WORKER = QuestionPolicy(per_minute=5, per_hour=60, per_day=300)
ADMIN = QuestionPolicy(per_minute=10, per_hour=0, per_day=0)


def usage(minute=0, hour=0, day=0, global_day=0, oldest_minute=None, oldest_hour=None, oldest_day=None):
    return QuotaUsage(minute, hour, day, oldest_minute, oldest_hour, oldest_day, global_day)


# --- sliding window ------------------------------------------------------------

def test_window_allows_up_to_the_limit_then_refuses():
    clock = Clock()
    limiter = SlidingWindowLimiter(3, 60, clock=clock)
    assert [limiter.hit("ip") for _ in range(3)] == [None, None, None]
    assert limiter.hit("ip") is not None


def test_window_reports_time_until_the_oldest_hit_expires():
    clock = Clock()
    limiter = SlidingWindowLimiter(2, 60, clock=clock)
    limiter.hit("k")
    clock.now += 20
    limiter.hit("k")
    clock.now += 10
    assert limiter.hit("k") == 30  # oldest hit (t=0) frees up at t=60; now is t=30


def test_window_frees_a_slot_exactly_when_the_oldest_hit_ages_out():
    clock = Clock()
    limiter = SlidingWindowLimiter(1, 60, clock=clock)
    limiter.hit("k")
    clock.now += 59
    assert limiter.hit("k") is not None
    clock.now += 1
    assert limiter.hit("k") is None


def test_refused_hits_do_not_extend_the_lockout():
    clock = Clock()
    limiter = SlidingWindowLimiter(1, 60, clock=clock)
    limiter.hit("k")
    for _ in range(50):
        clock.now += 1
        limiter.hit("k")
    clock.now = 1000 + 60
    assert limiter.hit("k") is None


def test_keys_are_independent():
    limiter = SlidingWindowLimiter(1, 60, clock=Clock())
    assert limiter.hit("a") is None
    assert limiter.hit("b") is None
    assert limiter.hit("a") is not None


def test_zero_limit_disables_the_window():
    limiter = SlidingWindowLimiter(0, 60, clock=Clock())
    assert all(limiter.hit("k") is None for _ in range(100))


def test_idle_keys_are_swept():
    clock = Clock()
    limiter = SlidingWindowLimiter(5, 60, clock=clock)
    limiter.SWEEP_EVERY = 10
    for i in range(9):
        limiter.hit(f"ip{i}")
    clock.now += 120
    limiter.hit("fresh")  # 10th call triggers the sweep
    assert set(limiter._hits) == {"fresh"}


# --- one question at a time -----------------------------------------------------

def test_second_concurrent_question_is_refused():
    guard = InFlightGuard(clock=Clock())
    guard.acquire(7)
    try:
        guard.acquire(7)
    except RateLimited as e:
        assert e.scope == "concurrent"
        assert "previous question" in e.user_message()
    else:
        raise AssertionError("second acquire should be refused")


def test_other_users_are_unaffected():
    guard = InFlightGuard(clock=Clock())
    guard.acquire(7)
    guard.acquire(8)


def test_hold_releases_on_exception():
    guard = InFlightGuard(clock=Clock())
    try:
        with guard.hold(7):
            raise ValueError("pipeline blew up")
    except ValueError:
        pass
    guard.acquire(7)


def test_slot_is_released_when_a_stream_is_abandoned():
    """A generator closed mid-stream (tab closed) gets GeneratorExit; the slot must still free."""
    guard = InFlightGuard(clock=Clock())

    def stream():
        with guard.hold(7):
            yield "partial"
            yield "more"

    gen = stream()
    next(gen)
    gen.close()
    guard.acquire(7)


def test_stale_lease_is_reclaimed():
    clock = Clock()
    guard = InFlightGuard(stale_after=180, clock=clock)
    guard.acquire(7)
    clock.now += 181
    guard.acquire(7)


# --- quotas ------------------------------------------------------------------------

def test_under_every_limit_is_allowed():
    assert evaluate_quota(usage(minute=4, hour=59, day=299), WORKER, 3000) is None


def test_per_minute_limit_with_retry_time():
    refusal = evaluate_quota(usage(minute=5, hour=5, day=5, oldest_minute=45), WORKER, 3000)
    assert refusal.scope == "user_minute"
    assert refusal.retry_after == 15
    assert refusal.user_message() == "You've reached the limit of 5 questions per minute. Try again in 15 seconds."


def test_per_hour_and_per_day_limits():
    assert evaluate_quota(usage(hour=60, day=60, oldest_hour=600), WORKER, 3000).scope == "user_hour"
    assert evaluate_quota(usage(day=300, oldest_day=3600), WORKER, 3000).scope == "user_day"


def test_zero_disables_a_window_for_admins():
    assert evaluate_quota(usage(minute=1, hour=500, day=5000), ADMIN, 0) is None
    assert evaluate_quota(usage(minute=10, oldest_minute=1), ADMIN, 0).scope == "user_minute"


def test_global_budget_trips_before_personal_limits():
    refusal = evaluate_quota(usage(global_day=3000), WORKER, 3000)
    assert refusal.scope == "global_day"
    assert "whole help desk" in refusal.user_message()


def test_wait_formatting():
    assert format_wait(0.2) == "1 second"
    assert format_wait(40) == "40 seconds"
    assert format_wait(61) == "2 minutes"
    assert format_wait(3600) == "1 hour"
    assert format_wait(5000) == "2 hours"


if __name__ == "__main__":
    sys.exit(run_tests(dict(globals())))
