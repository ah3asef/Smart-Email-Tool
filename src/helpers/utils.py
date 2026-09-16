from __future__ import annotations

import html
import re
import unicodedata
from html.parser import HTMLParser
from typing import List


class _HTMLToTextParser(HTMLParser):
    """
    Minimal, dependency-free HTML-to-text converter.

    - Strips all tags, decodes HTML entities.
    - Drops the content of <script> and <style> elements entirely.
    - Drops the content of <blockquote> elements, since in HTML email
      these almost always wrap quoted reply history rather than the
      author's own message.
    - Inserts newlines at block-level tag boundaries so paragraphs and
      list items don't get glued together into one run-on line.
    """

    _BLOCK_TAGS = {
        "p", "div", "br", "li", "tr", "table",
        "h1", "h2", "h3", "h4", "h5", "h6",
    }
    _SKIP_CONTENT_TAGS = {"script", "style", "blockquote"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: List[str] = []
        self._skip_depth = 0
        # HTML often splits a visually continuous phrase across inline
        # elements (for styling or tracking).  Keep track of that boundary
        # so handle_data can preserve a word separator when the markup is
        # removed, without adding spaces before punctuation.
        self._pending_text_boundary = False

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ARG002
        if tag in self._SKIP_CONTENT_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth == 0:
            if tag in self._BLOCK_TAGS:
                self._chunks.append("\n")
            else:
                self._pending_text_boundary = True

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_CONTENT_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if self._skip_depth == 0:
            if tag in self._BLOCK_TAGS:
                self._chunks.append("\n")
            else:
                self._pending_text_boundary = True

    def handle_data(self, data: str) -> None:
        if self._skip_depth != 0 or not data:
            return

        if (
            self._pending_text_boundary
            and self._chunks
            and self._is_word_character(self._chunks[-1][-1])
            and self._is_word_character(data[0])
        ):
            self._chunks.append(" ")

        self._chunks.append(data)
        self._pending_text_boundary = False

    @staticmethod
    def _is_word_character(char: str) -> bool:
        """Return whether ``char`` is a letter, number, or combining mark.

        Unicode categories make this work for Arabic and mixed-script text,
        not just ASCII words.
        """
        return unicodedata.category(char)[0] in {"L", "N", "M"}

    def get_text(self) -> str:
        return "".join(self._chunks)


def html_to_text(raw_html: str) -> str:
    """Convert an HTML string to plain text (tags/scripts/styles/blockquotes stripped)."""
    if not raw_html:
        return ""
    parser = _HTMLToTextParser()
    parser.feed(raw_html)
    parser.close()
    return html.unescape(parser.get_text())


def looks_like_html(text: str) -> bool:
    """Heuristic check for whether a string contains HTML markup."""
    if not text:
        return False
    return bool(re.search(r"<\s*[a-zA-Z][^>]*>", text))


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs into one space, and trim trailing whitespace per line."""
    if not text:
        return ""
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines)


def strip_empty_lines(text: str, max_consecutive: int = 1) -> str:
    """Collapse runs of blank lines down to at most `max_consecutive`, and trim the edges."""
    if not text:
        return ""

    lines = text.split("\n")
    result: List[str] = []
    blank_run = 0

    for line in lines:
        if line.strip() == "":
            blank_run += 1
            if blank_run <= max_consecutive:
                result.append("")
        else:
            blank_run = 0
            result.append(line)

    return "\n".join(result).strip()


_URL_PATTERN = re.compile(r"\b(?:https?://|www\.)[^\s<>\"']+")


def remove_urls(text: str) -> str:
    """Remove URL strings (tracking links, unsubscribe pages, etc.) while keeping label text."""
    if not text:
        return ""
    return _URL_PATTERN.sub(" ", text)


_INVISIBLE_CHARACTERS_PATTERN = re.compile(
    r"[\u00ad\u200b\u200c\u200d\u200e\u200f\u2060\u206f\u00a0\u2007"
    r"\u2028\u2029\u0000-\u0008\u000b\u000c\u000e-\u001f]"
)


def remove_invisible_characters(text: str) -> str:
    """Replace invisible/spam-padding Unicode characters with a normal space."""
    if not text:
        return ""
    return _INVISIBLE_CHARACTERS_PATTERN.sub(" ", text)


_BRACKETED_LABEL_PATTERN = re.compile(r"\[[^\[\]\n]{1,50}\]")


def remove_bracketed_labels(text: str) -> str:
    """Drop short bracketed image/placeholder labels such as '[Coursera Logo]'."""
    if not text:
        return ""
    return _BRACKETED_LABEL_PATTERN.sub("", text)


_EMPTY_PAREMS_PATTERN = re.compile(r"\(\s*\)")


def remove_empty_parens(text: str) -> str:
    """Remove empty parentheses left behind after URL removal."""
    if not text:
        return ""
    return _EMPTY_PAREMS_PATTERN.sub("", text)
