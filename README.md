# Smart Email Tool

A RAG pipeline for email: ingest emails from Gmail, clean them, split into chunks, embed them, and store the vectors for retrieval.

---

## Email Cleaning Architecture

The cleaning stage sits at the start of the preprocessing pipeline:

`GmailLoader (providers/) -> EmailCleaner (preprocessing/cleaner/) -> EmailChunker (preprocessing/chunks/) -> EmbeddingService (preprocessing/embeddings/) -> Store`

### Design principles

- **Two-layer split:** `helpers/utils.py` holds *generic* text utilities (usable by any document type), while `preprocessing/cleaner/EmailCleaner.py` holds *email-specific* heuristics. A future `PdfCleaner` or `SlackMessageCleaner` reuses the utils but defines its own patterns.
- **Immutability:** `EmailCleaner.clean()` returns a new `Email` via `model_copy()`; the original object is never mutated and all fields other than `body` are preserved.
- **Ordered pipeline:** HTML stripping runs first (so text heuristics see plain text), then reply-history truncation, then signature removal, then whitespace normalization.

### Where cleaning runs

| Stage | Responsibility | Location |
| --- | --- | --- |
| HTML-to-text | Strip tags, drop `<script>`/`<style>`/`<blockquote>`, decode entities | `helpers/utils.py` |
| Reply/forward removal | Truncate body at first quoted/forwarded block marker | `EmailCleaner` |
| Quote-line removal | Drop standalone `> ` quoted lines | `EmailCleaner` |
| Signature removal | Truncate at sign-off phrase or RFC 3676 `-- ` delimiter | `EmailCleaner` |
| Normalization | Collapse duplicate spaces, trim per-line whitespace, collapse blank lines | `helpers/utils.py` + `EmailCleaner` |

### Cleaning scenarios handled by `EmailCleaner`

1. **HTML bodies** — raw `<html>` content is converted to plain text; tags and HTML entities removed.
2. **Embedded scripts/styles** — `<script>` and `<style>` blocks are dropped entirely.
3. **Quoted reply history (marker-based)** — truncates at the earliest of:
   - `On <date> ... wrote:` reply lines
   - Outlook `-----Original Message-----`
   - Outlook forwarded headers (`From: ... / Sent: ... / To: ...`)
   - `---------- Forwarded message ---------`
   - `Begin forwarded message:`
4. **Blockquoted replies in HTML** — `<blockquote>` content (typically quoted history) is dropped during HTML conversion.
5. **Inline quote markers** — standalone lines beginning with `>` are removed.
6. **Signatures** — truncates at a whole-line sign-off (`Best regards`, `Kind regards`, `Thanks`, `Regards`, `Sincerely`, `Cheers`, etc.) or at a `-- ` RFC 3676 delimiter.
7. **Whitespace normalization** — duplicate spaces/tabs collapsed, trailing whitespace trimmed per line.
8. **Blank-line collapse** — runs of empty lines reduced to a single blank line; leading/trailing blank space trimmed.