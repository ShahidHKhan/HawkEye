"""
The app's shared Postgres (Supabase) connection pool.

Lives in its own module so code that only needs the database -- accounts,
auditing, rate limiting, scripts/manage_users.py -- can import it without also
importing implementation/answer.py, which builds a Gemini client at import time
and refuses to load without GOOGLE_API_KEY.

refresh_kb.py / implementation/diff_articles.py and implementation/ingest.py keep
their own pools on purpose: they run as separate processes (GitHub Actions, a
local CLI), never inside the app.
"""
import os

from dotenv import load_dotenv
from psycopg_pool import ConnectionPool

load_dotenv(override=True)

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
if not SUPABASE_DB_URL:
    raise RuntimeError("SUPABASE_DB_URL not set — add it to your .env file")

db_pool = ConnectionPool(
    SUPABASE_DB_URL,
    min_size=1,
    max_size=5,
    check=ConnectionPool.check_connection,
)
