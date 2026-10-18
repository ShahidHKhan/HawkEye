"""
Audit trail: who signed in, who got locked out or rate limited, who changed
which account. Written to the audit_events table (see
supabase/migrations/2026-10-18_accounts.sql) and read back by the per-username
login lockout in implementation/accounts.py and by the admin Activity tab.

Two rules this module exists to enforce:

1. Writing an audit row never breaks the thing being audited. record_event()
   logs and swallows its own failures -- a Supabase hiccup must not turn into
   a failed login or a lost answer.
2. The client IP comes from one place. resolve_client_ip() is used both by the
   HTTP middleware in app.py (which only sees raw ASGI headers) and by Gradio
   event handlers (which see a gr.Request), so the two can't disagree.
"""
import ipaddress
import logging
import os
from contextvars import ContextVar
from typing import Mapping

from psycopg.types.json import Jsonb

from implementation.db import db_pool

log = logging.getLogger(__name__)

# Set per HTTP request by app.py's ClientContextMiddleware. Gradio's login route
# calls our auth function inside that request, so authenticate() can read the
# caller's IP from here even though Gradio only passes it (username, password).
client_ip: ContextVar[str | None] = ContextVar("client_ip", default=None)
user_agent: ContextVar[str | None] = ContextVar("user_agent", default=None)

USER_AGENT_MAX_CHARS = 300


def _valid_ip(value: str | None) -> str | None:
    """audit_events.ip is inet, so anything unparseable is dropped rather than failing the insert."""
    if not value:
        return None
    try:
        return str(ipaddress.ip_address(value.strip()))
    except ValueError:
        return None


def resolve_client_ip(headers: Mapping[str, str], peer: str | None) -> str | None:
    """
    The real browser IP. On Fly.io every request arrives from Fly's proxy, so the
    socket peer is useless; Fly sets Fly-Client-IP to the actual client and
    overwrites any value the client sent, which makes it safe to trust *there*.
    Off Fly (local dev) the header could be forged by anyone, so it's ignored and
    the socket peer is used instead. FLY_APP_NAME is set on every Fly machine.
    `headers` must have lowercase keys.
    """
    if os.getenv("FLY_APP_NAME"):
        fly_ip = _valid_ip(headers.get("fly-client-ip"))
        if fly_ip:
            return fly_ip
    return _valid_ip(peer)


def request_context(request) -> tuple[str | None, str | None]:
    """(ip, user_agent) for a gr.Request, or (None, None) when there isn't one."""
    if request is None:
        return None, None
    headers = {k.lower(): v for k, v in dict(getattr(request, "headers", {}) or {}).items()}
    client = getattr(request, "client", None)
    peer = getattr(client, "host", None) if client is not None else None
    return resolve_client_ip(headers, peer), headers.get("user-agent")


def record_event(
    event_type: str,
    *,
    user_id: int | None = None,
    username: str | None = None,
    details: dict | None = None,
    ip: str | None = None,
    agent: str | None = None,
) -> None:
    """
    Append one audit_events row. `username` goes in username_attempted: the name
    typed at the login form for login events, or the account acted on for admin
    actions (the acting admin is `user_id`). ip/agent default to the current HTTP
    request's, when called inside one.
    """
    ip = _valid_ip(ip) or client_ip.get()
    agent = (agent or user_agent.get() or None)
    if agent:
        agent = agent[:USER_AGENT_MAX_CHARS]
    try:
        with db_pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO audit_events (event_type, user_id, username_attempted, ip, user_agent, details)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    event_type,
                    user_id,
                    username[:64] if username else None,
                    ip,
                    agent,
                    Jsonb(details) if details is not None else None,
                ),
            )
    except Exception:
        log.exception("Failed to record audit event %s for %s", event_type, username or user_id)
