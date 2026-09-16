"""Formatting of retrieved email chunks for an LLM prompt."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping


class ContextBuilder:
    """Turn retrieved chunks into readable source-grounded context."""

    def build(self, results: Iterable[Mapping[str, Any]]) -> str:
        """Return LLM-ready context, with a separator between source chunks."""
        grouped_results: Dict[str, List[Mapping[str, Any]]] = {}
        for index, result in enumerate(results or []):
            metadata = result.get("metadata") or {}
            if not isinstance(metadata, Mapping):
                metadata = {}

            email_id = str(metadata.get("email_id") or result.get("email_id") or f"result-{index}")
            grouped_results.setdefault(email_id, []).append(result)

        sections = []
        for email_number, chunks in enumerate(grouped_results.values(), start=1):
            first_chunk = chunks[0]
            metadata = first_chunk.get("metadata") or {}
            if not isinstance(metadata, Mapping):
                metadata = {}

            lines = [f"[EMAIL {email_number}]"]
            for label, key in (("Subject", "subject"), ("Sender", "sender"), ("Date", "date")):
                value = metadata.get(key) or first_chunk.get(key)
                if value:
                    lines.append(f"{label}: {value}")

            seen_texts = set()
            chunk_texts = []
            for chunk in chunks:
                text = str(chunk.get("text") or "").strip()
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    chunk_texts.append(text)

            if chunk_texts:
                if lines:
                    lines.append("")
                lines.extend(chunk_texts)

            if lines:
                sections.append("\n".join(lines))

        return "\n\n---\n\n".join(sections)
