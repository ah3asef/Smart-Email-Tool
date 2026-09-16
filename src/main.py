# Very simple pipeline for testing

from pathlib import Path

from providers.GmailProvider import GmailLoader
from preprocessing.cleaner.EmailCleaner import EmailCleaner
from preprocessing.chunks.email_chunking import EmailChunker
from preprocessing.embeddings.embedding_service import EmbeddingService
from chroma_storage.chroma_storage import ChromaStore

TOKEN_PATH = Path(__file__).resolve().parent.parent / "secrets" / "token_gmail.json"
Chunker = EmailChunker()

def main() -> None:
    # loader = GmailLoader(token_path=str(TOKEN_PATH), max_emails=10, label="INBOX")

    # with loader:
    #     emails = loader.fetch_emails()

    # print(f"Fetched {len(emails)} emails\n")
    # for email in emails:
    #     preview = email.body.replace("\n", " ")[:80]
    #     print(f"- {email.date.isoformat()} | {email.subject}\n"
    #           f"  from: {email.sender}\n"
    #           f"  preview: {preview}\n")
        
    #      # 2. Chunk emails
    # chunker = EmailChunker()
    # chunks = chunker.chunk_emails(emails)
    # print(f"Created {len(chunks)} chunks")

    # # 3. Create embeddings
    # embedding_service = EmbeddingService()
    # texts = [chunk["text"] for chunk in chunks]
    # embeddings = embedding_service.embed_texts(texts)
    # print("Embeddings created")

    # 4. Store in ChromaDB
    store = ChromaStore()
    # store.add(chunks, embeddings)
    # print("Stored in ChromaDB successfully")
    query = "Search about devops"
    print(store.similarity_search(query))
    


if __name__ == "__main__":
    main()