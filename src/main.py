"""Smoke test for GmailLoader: connect, fetch emails, print a summary."""

from pathlib import Path

from providers.GmailProvider import GmailLoader

TOKEN_PATH = Path(__file__).resolve().parent.parent / "secrets" / "token_gmail.json"


def main() -> None:
    loader = GmailLoader(token_path=str(TOKEN_PATH), max_emails=10, label="INBOX")

    with loader:
        emails = loader.fetch_emails()

    print(f"Fetched {len(emails)} emails\n")
    for email in emails:
        preview = email.body.replace("\n", " ")[:80]
        print(f"- {email.date.isoformat()} | {email.subject}\n"
              f"  from: {email.sender}\n"
              f"  preview: {preview}\n")


if __name__ == "__main__":
    main()