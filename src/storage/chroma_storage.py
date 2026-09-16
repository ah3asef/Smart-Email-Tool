import chromadb
from helpers.config import get_settings

class ChromaStore:

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=get_settings().persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=get_settings().collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks, embeddings):

        self.collection.add(
            ids=[chunk["chunk_id"] for chunk in chunks],

            documents=[
                chunk["text"]
                for chunk in chunks
            ],

            embeddings=embeddings,

            metadatas=[
                {
                    "email_id": chunk["email_id"],
                    "subject": chunk["subject"],
                    "sender": chunk["sender"],
                    "date": chunk["date"],
                    "chunk_index": chunk["chunk_index"]
                }
                for chunk in chunks
            ]
        )
    def search(self, query_embedding: list[float], top_k: int = 3):

        return self.collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        )
    
    def similarity_search(self, query: str, k: int = 5) :

        results = self.collection.query(query_texts=[query], n_results=k)

        if not results["ids"] or not results["ids"][0]:
            return []

        chunks = []
        for i in range(len(results["ids"][0])):
            doc_id = results["ids"][0][i]
            text = results["documents"][0][i]
            meta = dict(results["metadatas"][0][i]) if results["metadatas"] else {}
            chunk_id = meta.pop("_id", doc_id)
            chunks.append({
                "id": chunk_id,
                "email_id": meta.get("email_id", ""),
                "text": text,
                "chunk_index": meta.get("chunk_index", 0),
                "metadata": meta,
            })

        return chunks
    