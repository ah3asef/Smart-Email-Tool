import chromadb


class ChromaStore:

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path="./chroma_db"
        )

        self.collection = self.client.get_or_create_collection(
            name="emails"
        )

    def add_chunks(self, chunks, embeddings):

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