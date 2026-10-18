"""
HawkEye accounts: one row per technician in app_users, two roles.

    admin  -- every tab: Assistant, Refresh History, Knowledge Map, Accounts, Activity
    worker -- the Assistant tab only

Usernames are the part of a SUNY New Paltz address before @newpaltz.edu
(khans45 -> khans45@newpaltz.edu). The login form accepts either form, in any
case; normalize_username() is the single place that decides what counts.

Passwords: every account stores its own argon2id hash. During the account-testing
phase each account is created with its role's shared default password, read from
WORKER_DEFAULT_PASSWORD / ADMIN_DEFAULT_PASSWORD (Fly secrets, never the repo).
Because the admin default differs from the worker default, knowing the worker
password doesn't let anyone sign in as an admin -- and changing someone's role
resets their password to the new role's default for the same reason.

Nothing here imports Gradio: app.py turns these results into UI, and
scripts/manage_users.py drives the same functions from the command line.
"""
import logging
import os
import re
import threading
import time
from dataclasses import dataclass

import anyio
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from implementation import audit
from implementation.db import db_pool

log = logging.getLogger(__name__)

EMAIL_DOMAIN = "newpaltz.edu"
USERNAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{1,31}$")
ROLES = ("admin", "worker")
ROLE_PASSWORD_ENV = {"worker": "WORKER_DEFAULT_PASSWORD", "admin": "ADMIN_DEFAULT_PASSWORD"}

# OWASP's recommended argon2id floor (19 MiB, 2 passes, 1 lane) rather than
# argon2-cffi's 64 MiB default: the app runs on a 1 GB shared-CPU Fly machine, and
# a burst of logins at the default would cost 64 MiB apiece. Hashes record their
# own parameters, so raising these later only needs check_needs_rehash -- already
# wired into authenticate().
_hasher = PasswordHasher(time_cost=2, memory_cost=19456, parallelism=1)

# At most two hashes computed at once, whatever the login traffic: keeps a login
# flood from eating the machine's CPU and memory, and slows brute force further.
_verify_slots = threading.BoundedSemaphore(2)

# Verified against when the username doesn't exist, so a wrong username takes as
# long as a wrong password and response time can't reveal who has an account.
_DUMMY_HASH = _hasher.hash("hawkeye-timing-equalizer")

LOGIN_MAX_FAILURES = int(os.getenv("LOGIN_MAX_FAILURES", 5))
LOGIN_LOCKOUT_SECONDS = int(os.getenv("LOGIN_LOCKOUT_SECONDS", 15 * 60))

# How long a looked-up account is trusted before re-reading app_users. This is
# also the most time a disabled account keeps working in an already-open tab.
USER_CACHE_SECONDS = 30


class AccountError(Exception):
    """A refused account operation. str(error) is safe to show an admin as-is."""


@dataclass(frozen=True)
class User:
    id: int
    username: str
    role: str
    is_active: bool

    @property
    def email(self) -> str:
        return f"{self.username}@{EMAIL_DOMAIN}"

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


# ---------------------------------------------------------------------------
# Usernames and passwords
# ---------------------------------------------------------------------------

def normalize_username(raw: str | None) -> str | None:
    """
    'KhanS45', ' khans45 ', 'khans45@newpaltz.edu' -> 'khans45'. None if what's
    left isn't a plausible New Paltz username (wrong domain, spaces, symbols...).
    """
    if not raw:
        return None
    name = raw.strip().lower()
    suffix = "@" + EMAIL_DOMAIN
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return name if USERNAME_PATTERN.fullmatch(name) else None


def default_password(role: str) -> str:
    if role not in ROLES:
        raise AccountError(f"Unknown role {role!r}.")
    value = os.getenv(ROLE_PASSWORD_ENV[role])
    if not value:
        raise RuntimeError(f"{ROLE_PASSWORD_ENV[role]} is not set.")
    return value


def require_default_passwords() -> None:
    """
    Refuse to start without both default passwords, or with them equal (equal
    would hand admin access to anyone who knows the worker password).
    """
    missing = [env for env in ROLE_PASSWORD_ENV.values() if not os.getenv(env)]
    if missing:
        raise RuntimeError(
            f"{' and '.join(missing)} not set -- refusing to start. Add them to your "
            f".env file (see .env.example), or to the host's secrets when deploying."
        )
    if os.getenv("WORKER_DEFAULT_PASSWORD") == os.getenv("ADMIN_DEFAULT_PASSWORD"):
        raise RuntimeError(
            "WORKER_DEFAULT_PASSWORD and ADMIN_DEFAULT_PASSWORD are the same -- any worker "
            "could sign in as an admin. Give admins a different password."
        )


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    with _verify_slots:
        try:
            return _hasher.verify(password_hash, password)
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            return False


# ---------------------------------------------------------------------------
# Looking accounts up
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, User | None]] = {}
_cache_lock = threading.Lock()


def _row_to_user(row) -> User:
    return User(id=row[0], username=row[1], role=row[2], is_active=row[3])


def fetch_user(username: str) -> User | None:
    """Uncached lookup by already-normalized username."""
    with db_pool.connection() as conn:
        row = conn.execute(
            "SELECT id, username, role, is_active FROM app_users WHERE username = %s",
            (username,),
        ).fetchone()
    return _row_to_user(row) if row else None


def get_user(username: str) -> User | None:
    """Lookup by already-normalized username, cached for USER_CACHE_SECONDS."""
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(username)
        if hit and hit[0] > now:
            return hit[1]
    user = fetch_user(username)
    with _cache_lock:
        _cache[username] = (now + USER_CACHE_SECONDS, user)
    return user


def invalidate(username: str | None = None) -> None:
    with _cache_lock:
        if username is None:
            _cache.clear()
        else:
            _cache.pop(username, None)


def current_user(request) -> User | None:
    """
    The signed-in, still-active account behind a gr.Request, or None.

    Gradio hands handlers the username exactly as it was typed at the login form
    ('KhanS45@newpaltz.edu'), so it is normalized here on every call rather than
    trusted. An account disabled after sign-in stops resolving within
    USER_CACHE_SECONDS, even though Gradio's own session token lives on.
    """
    name = normalize_username(getattr(request, "username", None) if request is not None else None)
    if name is None:
        return None
    user = get_user(name)
    return user if user is not None and user.is_active else None


def has_active_admin() -> bool:
    with db_pool.connection() as conn:
        row = conn.execute(
            "SELECT EXISTS (SELECT 1 FROM app_users WHERE role = 'admin' AND is_active)"
        ).fetchone()
    return bool(row[0])


# ---------------------------------------------------------------------------
# Signing in
# ---------------------------------------------------------------------------

def recent_login_failures(username: str) -> int:
    """
    Failed sign-ins for this username inside the lockout window, counting only
    those since its last successful sign-in or admin password reset -- so an
    admin resetting a locked-out technician's password also unlocks them.
    """
    with db_pool.connection() as conn:
        row = conn.execute(
            """
            SELECT count(*) FROM audit_events
            WHERE event_type = 'login_failure'
              AND username_attempted = %(u)s
              AND occurred_at > greatest(
                  now() - make_interval(secs => %(window)s),
                  coalesce(
                      (SELECT max(occurred_at) FROM audit_events
                       WHERE event_type IN ('login_success', 'password_reset')
                         AND username_attempted = %(u)s),
                      '-infinity'::timestamptz
                  )
              )
            """,
            {"u": username, "window": LOGIN_LOCKOUT_SECONDS},
        ).fetchone()
    return int(row[0])


def authenticate_sync(raw_username: str, password: str) -> bool:
    """
    Gradio's login check. Every outcome is audited; the browser only ever learns
    "incorrect credentials", whichever of these was the actual reason.
    """
    typed = (raw_username or "").strip()[:64]
    name = normalize_username(raw_username)
    if name is None:
        verify_password(_DUMMY_HASH, password or "")
        audit.record_event("login_failure", username=typed, details={"reason": "invalid_username"})
        return False

    if recent_login_failures(name) >= LOGIN_MAX_FAILURES:
        audit.record_event(
            "login_locked", username=name,
            details={"max_failures": LOGIN_MAX_FAILURES, "window_seconds": LOGIN_LOCKOUT_SECONDS},
        )
        return False

    with db_pool.connection() as conn:
        row = conn.execute(
            "SELECT id, username, role, is_active, password_hash FROM app_users WHERE username = %s",
            (name,),
        ).fetchone()

    password_ok = verify_password(row[4] if row else _DUMMY_HASH, password or "")
    if row is None or not row[3] or not password_ok:
        reason = "unknown_user" if row is None else ("disabled" if not row[3] else "bad_password")
        audit.record_event(
            "login_failure", user_id=row[0] if row else None, username=name,
            details={"reason": reason, "typed": typed} if typed != name else {"reason": reason},
        )
        return False

    user_id, stored_hash = row[0], row[4]
    with db_pool.connection() as conn:
        if _hasher.check_needs_rehash(stored_hash):
            conn.execute(
                "UPDATE app_users SET password_hash = %s WHERE id = %s",
                (hash_password(password), user_id),
            )
        conn.execute("UPDATE app_users SET last_login_at = now() WHERE id = %s", (user_id,))
    invalidate(name)
    audit.record_event("login_success", user_id=user_id, username=name)
    return True


async def authenticate(username: str, password: str) -> bool:
    """
    Async so Gradio awaits it instead of running a ~50 ms argon2 hash plus three
    database round trips on the event loop, where it would stall every other
    user's stream. anyio copies the current context into the worker thread, so
    audit.client_ip set by the middleware is still visible inside.
    """
    try:
        return await anyio.to_thread.run_sync(authenticate_sync, username, password)
    except Exception:
        log.exception("Login check failed for %r", username)
        return False


# ---------------------------------------------------------------------------
# Admin operations (the Accounts tab and scripts/manage_users.py)
# ---------------------------------------------------------------------------

def _require_valid_username(raw: str) -> str:
    name = normalize_username(raw)
    if name is None:
        raise AccountError(
            f"'{(raw or '').strip()}' isn't a valid New Paltz username. Use the part before "
            f"@{EMAIL_DOMAIN}: 2-32 characters, letters, numbers, dots, dashes or underscores."
        )
    return name


def create_user(actor: User | None, raw_username: str, role: str = "worker") -> User:
    """Provision an account with its role's default password. actor None = the CLI."""
    name = _require_valid_username(raw_username)
    if role not in ROLES:
        raise AccountError(f"Unknown role {role!r}.")
    password_hash = hash_password(default_password(role))
    with db_pool.connection() as conn:
        row = conn.execute(
            """
            INSERT INTO app_users (username, role, password_hash, created_by)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (username) DO NOTHING
            RETURNING id, username, role, is_active
            """,
            (name, role, password_hash, actor.id if actor else None),
        ).fetchone()
    if row is None:
        raise AccountError(f"{name} already has an account (if it's disabled, enable it below).")
    invalidate(name)
    audit.record_event(
        "account_created", user_id=actor.id if actor else None, username=name,
        details={"role": role, "by": actor.username if actor else "cli"},
    )
    return _row_to_user(row)


def bulk_create(actor: User, text: str, role: str = "worker",
                budget_check=None) -> list[tuple[str, bool, str]]:
    """
    One username (or address) per line; blank lines and duplicates skipped.
    Returns (line, created?, message) per line so the admin sees exactly which
    entries failed and why, instead of one error aborting the batch.
    `budget_check`, if given, runs before each creation and raises AccountError
    to refuse it (app.py uses it for the hourly account-creation limit).
    """
    results, seen = [], set()
    for line in (text or "").splitlines():
        entry = line.strip().strip(",;")
        if not entry:
            continue
        key = normalize_username(entry) or entry
        if key in seen:
            continue
        seen.add(key)
        try:
            if budget_check is not None:
                budget_check()
            user = create_user(actor, entry, role)
            results.append((entry, True, f"created {user.username} ({user.email})"))
        except AccountError as e:
            results.append((entry, False, str(e)))
    return results


def _lock_target(conn, raw_username: str):
    name = _require_valid_username(raw_username)
    row = conn.execute(
        "SELECT id, username, role, is_active FROM app_users WHERE username = %s FOR UPDATE",
        (name,),
    ).fetchone()
    if row is None:
        raise AccountError(f"No account named {name}.")
    return _row_to_user(row)


def check_guardrails(actor: User, target: User, *, deactivating: bool, demoting: bool,
                     other_active_admins: int) -> None:
    """Pure rule check, split out so the rules are testable without a database."""
    if actor is not None and actor.id == target.id and (deactivating or demoting):
        raise AccountError("You can't disable or demote your own account.")
    if target.is_admin and target.is_active and (deactivating or demoting) and other_active_admins == 0:
        raise AccountError(f"{target.username} is the only active admin; promote someone else first.")


def _other_active_admins(conn, target: User) -> int:
    # Lock every active admin row so two admins demoting each other at once
    # can't both pass the check and leave zero admins.
    conn.execute("SELECT id FROM app_users WHERE role = 'admin' AND is_active FOR UPDATE")
    return conn.execute(
        "SELECT count(*) FROM app_users WHERE role = 'admin' AND is_active AND id <> %s",
        (target.id,),
    ).fetchone()[0]


def set_active(actor: User, raw_username: str, active: bool) -> User:
    with db_pool.connection() as conn:
        target = _lock_target(conn, raw_username)
        if target.is_active == active:
            raise AccountError(f"{target.username} is already {'active' if active else 'disabled'}.")
        if not active:
            check_guardrails(actor, target, deactivating=True, demoting=False,
                             other_active_admins=_other_active_admins(conn, target))
        conn.execute("UPDATE app_users SET is_active = %s WHERE id = %s", (active, target.id))
    invalidate(target.username)
    audit.record_event(
        "account_enabled" if active else "account_disabled",
        user_id=actor.id, username=target.username, details={"by": actor.username},
    )
    return User(target.id, target.username, target.role, active)


def set_role(actor: User, raw_username: str, role: str) -> User:
    """Change role and reset the password to the new role's default (see module docstring)."""
    if role not in ROLES:
        raise AccountError(f"Unknown role {role!r}.")
    new_hash = hash_password(default_password(role))
    with db_pool.connection() as conn:
        target = _lock_target(conn, raw_username)
        if target.role == role:
            raise AccountError(f"{target.username} is already {'an' if role == 'admin' else 'a'} {role}.")
        demoting = target.is_admin and role != "admin"
        if demoting:
            check_guardrails(actor, target, deactivating=False, demoting=True,
                             other_active_admins=_other_active_admins(conn, target))
        conn.execute(
            "UPDATE app_users SET role = %s, password_hash = %s, must_change_password = true WHERE id = %s",
            (role, new_hash, target.id),
        )
    invalidate(target.username)
    audit.record_event(
        "role_changed", user_id=actor.id, username=target.username,
        details={"from": target.role, "to": role, "by": actor.username, "password_reset": True},
    )
    return User(target.id, target.username, role, target.is_active)


def reset_password(actor: User | None, raw_username: str) -> User:
    """Back to the role's default password. Also clears a login lockout."""
    with db_pool.connection() as conn:
        target = _lock_target(conn, raw_username)
        conn.execute(
            "UPDATE app_users SET password_hash = %s, must_change_password = true WHERE id = %s",
            (hash_password(default_password(target.role)), target.id),
        )
    invalidate(target.username)
    audit.record_event(
        "password_reset", user_id=actor.id if actor else None, username=target.username,
        details={"by": actor.username if actor else "cli"},
    )
    return target


def bootstrap_admin(raw_username: str) -> tuple[User, str]:
    """
    Break-glass for scripts/manage_users.py: make this account an active admin
    with the admin default password, creating it if needed. Returns (user, what happened).
    """
    name = _require_valid_username(raw_username)
    password_hash = hash_password(default_password("admin"))
    with db_pool.connection() as conn:
        existing = conn.execute(
            "SELECT id FROM app_users WHERE username = %s FOR UPDATE", (name,)
        ).fetchone()
        row = conn.execute(
            """
            INSERT INTO app_users (username, role, password_hash, is_active)
            VALUES (%s, 'admin', %s, true)
            ON CONFLICT (username) DO UPDATE
                SET role = 'admin', password_hash = EXCLUDED.password_hash, is_active = true,
                    must_change_password = true
            RETURNING id, username, role, is_active
            """,
            (name, password_hash),
        ).fetchone()
    user = _row_to_user(row)
    invalidate(name)
    what = "repaired (admin, active, admin default password)" if existing else "created"
    audit.record_event(
        "account_created" if not existing else "password_reset",
        username=name, details={"by": "cli bootstrap", "role": "admin"},
    )
    return user, what


def list_users() -> list[tuple]:
    """Rows for the Accounts tab, newest first, from the v_user_usage view."""
    with db_pool.connection() as conn:
        return conn.execute(
            """
            SELECT username, email, role, is_active,
                   to_char(created_at AT TIME ZONE 'America/New_York', 'YYYY-MM-DD'),
                   to_char(last_login_at AT TIME ZONE 'America/New_York', 'YYYY-MM-DD HH24:MI'),
                   queries_7d,
                   to_char(last_query_at AT TIME ZONE 'America/New_York', 'YYYY-MM-DD HH24:MI')
            FROM v_user_usage
            ORDER BY is_active DESC, role, username
            """
        ).fetchall()
