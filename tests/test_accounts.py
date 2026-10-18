"""
Offline tests for implementation/accounts.py: username rules, password hashing,
the role guardrails, and the session lookup. No database -- anything that would
query one is stubbed.

    python tests/test_accounts.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://placeholder/unused")
os.environ.setdefault("GOOGLE_API_KEY", "placeholder-unused")

from implementation import accounts  # noqa: E402
from implementation.accounts import AccountError, User  # noqa: E402
from _runner import run_tests  # noqa: E402


class FakeRequest:
    def __init__(self, username):
        self.username = username


ADMIN = User(id=1, username="khans45", role="admin", is_active=True)
OTHER_ADMIN = User(id=2, username="boss1", role="admin", is_active=True)
WORKER = User(id=3, username="jdoe12", role="worker", is_active=True)


def raises(fn, exc=AccountError) -> str:
    try:
        fn()
    except exc as e:
        return str(e)
    raise AssertionError(f"expected {exc.__name__}")


# --- usernames ---------------------------------------------------------------

def test_plain_username_is_kept():
    assert accounts.normalize_username("khans45") == "khans45"


def test_case_and_whitespace_are_normalized():
    assert accounts.normalize_username("  KhanS45 ") == "khans45"


def test_full_new_paltz_address_maps_to_the_username():
    assert accounts.normalize_username("khans45@newpaltz.edu") == "khans45"
    assert accounts.normalize_username("KHANS45@NewPaltz.edu") == "khans45"


def test_other_domains_are_rejected():
    assert accounts.normalize_username("khans45@gmail.com") is None
    assert accounts.normalize_username("khans45@hawkmail.newpaltz.edu.evil.com") is None


def test_invalid_usernames_are_rejected():
    for bad in ["", None, "a", "has space", "semi;colon", "-leadingdash", "x" * 40, "drop'table"]:
        assert accounts.normalize_username(bad) is None, bad


def test_dots_dashes_underscores_are_allowed():
    assert accounts.normalize_username("j.doe-2_x") == "j.doe-2_x"


def test_email_is_derived_from_username():
    assert WORKER.email == "jdoe12@newpaltz.edu"


# --- passwords ---------------------------------------------------------------

def test_hash_round_trip():
    hashed = accounts.hash_password("1hawkdrive")
    assert hashed.startswith("$argon2id$")
    assert accounts.verify_password(hashed, "1hawkdrive")
    assert not accounts.verify_password(hashed, "1HAWKDRIVE")


def test_same_password_hashes_differently_per_account():
    assert accounts.hash_password("1hawkdrive") != accounts.hash_password("1hawkdrive")


def test_garbage_hash_fails_closed():
    assert not accounts.verify_password("not-a-hash", "anything")


def test_default_passwords_are_required():
    saved = {k: os.environ.pop(k, None) for k in accounts.ROLE_PASSWORD_ENV.values()}
    try:
        message = raises(accounts.require_default_passwords, RuntimeError)
        assert "WORKER_DEFAULT_PASSWORD" in message and "ADMIN_DEFAULT_PASSWORD" in message
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_equal_default_passwords_are_refused():
    saved = {k: os.environ.get(k) for k in accounts.ROLE_PASSWORD_ENV.values()}
    os.environ["WORKER_DEFAULT_PASSWORD"] = os.environ["ADMIN_DEFAULT_PASSWORD"] = "same"
    try:
        assert "same" in raises(accounts.require_default_passwords, RuntimeError)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --- guardrails --------------------------------------------------------------

def test_admin_cannot_disable_themselves():
    message = raises(lambda: accounts.check_guardrails(
        ADMIN, ADMIN, deactivating=True, demoting=False, other_active_admins=1))
    assert "your own account" in message


def test_admin_cannot_demote_themselves():
    raises(lambda: accounts.check_guardrails(
        ADMIN, ADMIN, deactivating=False, demoting=True, other_active_admins=3))


def test_last_admin_cannot_be_removed():
    message = raises(lambda: accounts.check_guardrails(
        ADMIN, OTHER_ADMIN, deactivating=True, demoting=False, other_active_admins=0))
    assert "only active admin" in message


def test_admin_can_disable_another_admin_when_one_remains():
    accounts.check_guardrails(ADMIN, OTHER_ADMIN, deactivating=True, demoting=False, other_active_admins=1)


def test_disabling_a_worker_is_always_allowed():
    accounts.check_guardrails(ADMIN, WORKER, deactivating=True, demoting=False, other_active_admins=0)


def test_bulk_create_reports_each_line_and_skips_blanks_and_duplicates():
    created = []

    def fake_create(actor, entry, role):
        name = accounts.normalize_username(entry)
        if name is None:
            raise AccountError(f"'{entry}' isn't a valid New Paltz username.")
        created.append(name)
        return User(id=10 + len(created), username=name, role=role, is_active=True)

    original = accounts.create_user
    accounts.create_user = fake_create
    try:
        results = accounts.bulk_create(ADMIN, "jdoe12\n\n  JDOE12@newpaltz.edu\nbad name\nasmith9,\n")
    finally:
        accounts.create_user = original
    assert [ok for _, ok, _ in results] == [True, False, True]
    assert created == ["jdoe12", "asmith9"]


def test_bulk_create_stops_creating_when_the_budget_runs_out():
    calls = {"n": 0}

    def budget():
        calls["n"] += 1
        if calls["n"] > 1:
            raise AccountError("hourly limit on new accounts reached")

    original = accounts.create_user
    accounts.create_user = lambda actor, entry, role: User(1, entry, role, True)
    try:
        results = accounts.bulk_create(ADMIN, "aaa\nbbb\nccc", budget_check=budget)
    finally:
        accounts.create_user = original
    assert [ok for _, ok, _ in results] == [True, False, False]
    assert "hourly limit" in results[1][2]


# --- session lookup ----------------------------------------------------------

def with_lookup(users: dict, fn):
    original = accounts.fetch_user
    accounts.fetch_user = lambda name: users.get(name)
    accounts.invalidate()
    try:
        return fn()
    finally:
        accounts.fetch_user = original
        accounts.invalidate()


def test_current_user_normalizes_what_was_typed_at_login():
    """Gradio passes the username exactly as typed; it must still resolve."""
    user = with_lookup({"khans45": ADMIN}, lambda: accounts.current_user(FakeRequest("KhanS45@newpaltz.edu")))
    assert user == ADMIN


def test_disabled_account_does_not_resolve():
    disabled = User(id=3, username="jdoe12", role="worker", is_active=False)
    assert with_lookup({"jdoe12": disabled}, lambda: accounts.current_user(FakeRequest("jdoe12"))) is None


def test_unknown_or_missing_user_does_not_resolve():
    assert with_lookup({}, lambda: accounts.current_user(FakeRequest("ghost99"))) is None
    assert with_lookup({}, lambda: accounts.current_user(FakeRequest(None))) is None
    assert with_lookup({}, lambda: accounts.current_user(None)) is None


def test_lookups_are_cached_until_invalidated():
    calls = []

    def fetch(name):
        calls.append(name)
        return WORKER

    original = accounts.fetch_user
    accounts.fetch_user = fetch
    accounts.invalidate()
    try:
        accounts.get_user("jdoe12")
        accounts.get_user("jdoe12")
        assert calls == ["jdoe12"]
        accounts.invalidate("jdoe12")
        accounts.get_user("jdoe12")
        assert calls == ["jdoe12", "jdoe12"]
    finally:
        accounts.fetch_user = original
        accounts.invalidate()


def test_invalid_username_login_is_refused_without_touching_the_database():
    events = []
    original = accounts.audit.record_event
    accounts.audit.record_event = lambda event_type, **kw: events.append((event_type, kw))
    try:
        assert accounts.authenticate_sync("not a username!", "1hawkdrive") is False
    finally:
        accounts.audit.record_event = original
    assert events[0][0] == "login_failure"
    assert events[0][1]["details"]["reason"] == "invalid_username"


if __name__ == "__main__":
    sys.exit(run_tests(dict(globals())))
