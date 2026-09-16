from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.EmailModel import Email


class EmailChunker:

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_email(self, email: Email):

        text = email.body or ""

        chunks = self.splitter.split_text(text)

        return [
            {
                "chunk_id": f"{email.id}_{index}",
                "email_id": email.id,
                "text": chunk,
                "subject": email.subject,
                "sender": str(email.sender),
                "recipients": [str(r) for r in email.recipients],
                "date": email.date.isoformat(),
                "chunk_index": index
            }
            for index, chunk in enumerate(chunks)
        ]

    def chunk_emails(self, emails: List[Email]):

        all_chunks = []

        for email in emails:
            chunks = self.chunk_email(email)
            all_chunks.extend(chunks)

        return all_chunks
