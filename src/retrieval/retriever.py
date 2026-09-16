# Install necessary packages
!pip install -qqq chromadb sentence-transformers

import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import uuid

# 1. Embedding model (must match Person 2's choice exactly)

MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim vectors
embedder = SentenceTransformer(MODEL_NAME)



# 2. ChromaDB setup (persistent, so data survives restarts)
client = chromadb.PersistentClient(path="./chroma_store")

collection = client.get_or_create_collection(
    name="emails",
    metadata={"hnsw:space": "cosine"}  # similarity metric — must match how embeddings behave
)


# 3. DUMMY DATA LOADER —

def load_dummy_data():
    """Insert a handful of fake email chunks so you can test retrieval logic
    without waiting for the real pipeline."""
    dummy_chunks = [
        "Your order #4521 has been shipped and will arrive in 3 days.",
        "فاتورتك الخاصة بالطلب رقم 4521 مرفقة في هذا الإيميل.",
        "We're sorry, your refund request was rejected due to policy violation.",
        "الرجاء تأكيد بريدك الإلكتروني لتفعيل حسابك الجديد.",
        "Your subscription will renew automatically on the 1st of next month.",
    ]

    dummy_metadata = [
        {"sender": "shipping@store.com", "date": "2026-08-01"},
        {"sender": "billing@store.com", "date": "2026-08-02"},
        {"sender": "support@store.com", "date": "2026-08-03"},
        {"sender": "accounts@store.com", "date": "2026-08-04"},
        {"sender": "billing@store.com", "date": "2026-08-05"},
    ]

    embeddings = embedder.encode(dummy_chunks).tolist()
    ids = [str(uuid.uuid4()) for _ in dummy_chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=dummy_chunks,
        metadatas=dummy_metadata,
    )
    print(f"Loaded {len(dummy_chunks)} dummy chunks into ChromaDB.")

# 4. Query cleaning

def clean_query(text: str) -> str:
    """
    IMPORTANT: This must eventually call the SAME cleaning function
    Person 1 uses on the chunks. This is just a placeholder so you
    can develop independently right now.
    """
    return text.strip()


# 5. THE MAIN FUNCTION

def retrieve(query: str, top_k: int = 3) -> list[dict]:
    cleaned_query = clean_query(query)
    query_vector = embedder.encode([cleaned_query]).tolist()

    results = collection.query(
        query_embeddings=query_vector,
        n_results=top_k,
    )

    output = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "chunk": doc,
            "metadata": meta,
            "score": 1 - distance,  # convert distance -> similarity score
        })

    return output

# 6. Quick manual test

if __name__ == "__main__":
    load_dummy_data()

    test_query = "where is my invoice for order 4521?"
    results = retrieve(test_query, top_k=3)

    print(f"\nQuery: {test_query}\n")
    for r in results:
        print(f"Score: {r['score']:.3f} | {r['chunk']} | {r['metadata']}")
