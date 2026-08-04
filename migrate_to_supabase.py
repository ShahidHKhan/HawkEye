import os
from pathlib import Path

import psycopg
from chromadb import PersistentClient
from dotenv import load_dotenv

load_dotenv(override=True)

DB_NAME = str(Path(__file__).parent / "preprocessed_db")
COLLECTION_NAME = "docs"
BATCH_SIZE = 500

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
if not SUPABASE_DB_URL:
    raise RuntimeError("SUPABASE_DB_URL not set — add it to your .env file first")


def fetch_all_chunks():
    """Pull every chunk (text + metadata + embedding) out of the local Chroma store."""
    chroma = PersistentClient(path=DB_NAME)
    collection = chroma.get_or_create_collection(COLLECTION_NAME)
    result = collection.get(include=["documents", "metadatas", "embeddings"])
    return list(zip(result["documents"], result["metadatas"], result["embeddings"]))


def embedding_to_vector_literal(embedding: list[float]) -> str:
    """Format a python float list as a pgvector text literal, e.g. '[0.1,0.2,...]'."""
    return "[" + ",".join(str(x) for x in embedding) + "]"


def main():
    conn = psycopg.connect(SUPABASE_DB_URL)
    cur = conn.cursor()

    # Safety check: don't silently duplicate rows if this has already been run.
    cur.execute("SELECT count(*) FROM chunks")
    existing_count = cur.fetchone()[0]
    if existing_count > 0:
        print(f"chunks table already has {existing_count:,} rows — aborting to avoid duplicates.")
        print("If you really want to re-run this, truncate the table first: TRUNCATE chunks;")
        cur.close()
        conn.close()
        return

    chunks = fetch_all_chunks()
    print(f"Loaded {len(chunks):,} chunks from preprocessed_db")

    rows = [
        (meta.get("source"), meta.get("type"), doc, embedding_to_vector_literal(emb))
        for doc, meta, emb in chunks
    ]

    inserted = 0
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        cur.executemany(
            "INSERT INTO chunks (source, type, page_content, embedding) "
            "VALUES (%s, %s, %s, %s::vector)",
            batch,
        )
        conn.commit()
        inserted += len(batch)
        print(f"  inserted {inserted:,}/{len(rows):,}")

    cur.execute("SELECT count(*) FROM chunks")
    total = cur.fetchone()[0]
    print(f"\nDone. chunks table now has {total:,} rows.")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
