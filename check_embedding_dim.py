from pathlib import Path
from chromadb import PersistentClient

DB_NAME = str(Path(__file__).parent / "preprocessed_db")
COLLECTION_NAME = "docs"

chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(COLLECTION_NAME)

sample = collection.get(limit=1, include=["embeddings"])
count = collection.count()

print("Embedding dimension:", len(sample["embeddings"][0]))
print("Total chunks:", count)
