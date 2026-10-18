"""
Break-glass account management from the command line.

Day to day, accounts are managed from the Accounts tab in the app. This script
is for when that isn't possible: creating the very first admin, or getting back
in after every admin is disabled or locked out.

    uv run python scripts/manage_users.py bootstrap khans45   # make/repair an admin
    uv run python scripts/manage_users.py add jdoe12          # add a worker
    uv run python scripts/manage_users.py add boss1 --role admin
    uv run python scripts/manage_users.py reset jdoe12        # back to the role's default password
    uv run python scripts/manage_users.py list

Needs SUPABASE_DB_URL plus WORKER_DEFAULT_PASSWORD / ADMIN_DEFAULT_PASSWORD in
.env -- the same values the deployed app uses, or the passwords set here won't
match what technicians are told. Every change is written to audit_events.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from implementation import accounts  # noqa: E402
from implementation.accounts import AccountError  # noqa: E402


def cmd_bootstrap(args) -> int:
    user, what = accounts.bootstrap_admin(args.username)
    print(f"{user.username} ({user.email}): {what}. Sign in with ADMIN_DEFAULT_PASSWORD.")
    return 0


def cmd_add(args) -> int:
    user = accounts.create_user(None, args.username, args.role)
    print(f"Created {user.username} ({user.email}) as {user.role}.")
    return 0


def cmd_reset(args) -> int:
    user = accounts.reset_password(None, args.username)
    print(f"Reset {user.username}'s password to the {user.role} default (also clears a lockout).")
    return 0


def cmd_list(args) -> int:
    rows = accounts.list_users()
    if not rows:
        print("No accounts yet. Start with: scripts/manage_users.py bootstrap <username>")
        return 0
    print(f"{'username':<20} {'role':<7} {'active':<7} {'last sign-in (ET)':<18} questions (7d)")
    for username, _email, role, active, _created, last_login, queries_7d, _last_query in rows:
        print(f"{username:<20} {role:<7} {'yes' if active else 'no':<7} {last_login or '-':<18} {queries_7d}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("bootstrap", help="create or repair an admin account (active, admin default password)")
    p.add_argument("username")
    p.set_defaults(fn=cmd_bootstrap)

    p = sub.add_parser("add", help="create an account with its role's default password")
    p.add_argument("username")
    p.add_argument("--role", choices=accounts.ROLES, default="worker")
    p.set_defaults(fn=cmd_add)

    p = sub.add_parser("reset", help="reset an account to its role's default password")
    p.add_argument("username")
    p.set_defaults(fn=cmd_reset)

    p = sub.add_parser("list", help="list accounts")
    p.set_defaults(fn=cmd_list)

    args = parser.parse_args()
    if args.command != "list":
        accounts.require_default_passwords()
    try:
        return args.fn(args)
    except AccountError as e:
        print(f"Refused: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
