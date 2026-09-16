# Very simple pipeline for testing

from pathlib import Path

#from providers.GmailProvider import GmailLoader
from preprocessing.cleaner.EmailCleaner import EmailCleaner
from preprocessing.chunks.email_chunking import EmailChunker
from preprocessing.embeddings.embedding_service import EmbeddingService
from storage.chroma_storage import ChromaStore
from applications.Retriever import Retriever
from services.GenerationService import GenerationService
from services.OllamaClient import OllamaClient
from services.PromptBuilder import PromptBuilder
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
    embedding_service = EmbeddingService()
    # texts = [chunk["text"] for chunk in chunks]
    # embeddings = embedding_service.embed_texts(texts)
    # print("Embeddings created")

    # 4. Store in ChromaDB
    store = ChromaStore()
    # store.add(chunks, embeddings)
    # print("Stored in ChromaDB successfully")
    retriever = Retriever(embedding_service, store)
    generation_service = GenerationService(
        retriever=retriever,
        prompt_builder=PromptBuilder(),
        llm_client=OllamaClient(),
    )

    query = "What emails did I receive about DevOps jobs?"
    result = generation_service.generate(
        query=query,
        mode="question_answering",
        top_k=5,
    )

    print("Answer:\n")
    print(result.answer)

    if result.sources:
        print("\nSources:")
        for source in result.sources:
            print(f"- {source.subject} ({source.email_id}, {source.chunk_id})")



if __name__ == "__main__":
    main()
