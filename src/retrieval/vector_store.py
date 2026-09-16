import chromadb

CHROMA_PATH = "./chroma_store"
COLLECTION_NAME = "emails"


def get_collection():
    """
    Returns a ChromaDB collection, creating it if it doesn't exist yet.
    Called by retriever.py — nothing outside this file should talk to
    ChromaDB directly.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def add_chunks(chunks: list[str], embeddings: list[list[float]], metadatas: list[dict], ids: list[str]):
    """
    Insert chunks + their embeddings + metadata into the vector store.
    Used both by your dummy-data loader now, and later by the real
    ingestion pipeline (Person 1 + Person 2's output).
    """
    collection = get_collection()
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )


def search(query_embedding: list[float], top_k: int = 3) -> dict:
    """
    Runs the actual similarity search. Returns ChromaDB's raw result
    format — retriever.py is responsible for reshaping this into the
    {"chunk", "metadata", "score"} format Person 4 expects.
    """
    collection = get_collection()
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
