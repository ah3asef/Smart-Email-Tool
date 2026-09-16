import re
import unicodedata
from typing import List, Optional, Pattern


from models.EmailModel import Email
from helpers.utils import (
    html_to_text,
    looks_like_html,
    normalize_whitespace,
    remove_bracketed_labels,
    remove_empty_parens,
    remove_invisible_characters,
    remove_urls,
    strip_empty_lines,
)


class EmailCleaner():
    """
    Cleans the body of an `Email`.

    All heuristics here are email-specific (unlike utils.py, which stays
    generic), so a future `SlackMessageCleaner` or `PdfCleaner` would
    define its own signature/quote patterns while still implementing the
    same `` contract.
    """

    # Lines that mark the start of a quoted/forwarded-reply block. The
    # earliest match of any of these truncates the rest of the body.
    _REPLY_HISTORY_PATTERNS: List[Pattern[str]] = [
        # "On Mon, Jul 20, 2026 at 9:00 AM, John Doe <john@example.com> wrote:"
        re.compile(r"^\s*On .{0,200}?wrote:\s*$", re.IGNORECASE | re.MULTILINE),
        # Outlook-style "-----Original Message-----"
        re.compile(r"^\s*-{2,}\s*Original Message\s*-{2,}\s*$", re.IGNORECASE | re.MULTILINE),
        # Outlook-style forwarded header block ("From: ... Sent: ... To: ...")
        re.compile(r"^\s*From:\s*.+\n\s*Sent:\s*.+\n\s*To:\s*.+", re.IGNORECASE | re.MULTILINE),
        # Gmail/Apple Mail forwarded message markers
        re.compile(r"^\s*-{2,}\s*Forwarded message\s*-{2,}\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*Begin forwarded message:\s*$", re.IGNORECASE | re.MULTILINE),
    ]

    # A line consisting only of quote markers, e.g. "> " or ">> some text"
    _QUOTE_LINE_PATTERN = re.compile(r"^\s*>.*$", re.MULTILINE)

    # Common sign-off phrases that typically precede a signature block.
    # Matched as a *whole line* so we don't clip "Thanks for the update."
    _SIGNOFF_PATTERN = re.compile(
        r"^\s*(best regards|kind regards|warm regards|regards|"
        r"best wishes|best|sincerely|many thanks|thanks so much|"
        r"thank you|thanks|cheers)\s*[,.!]?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    # Standard RFC 3676 signature delimiter: a line that is exactly "-- "
    _SIG_DELIMITER_PATTERN = re.compile(r"^-- ?\s*$", re.MULTILINE)

    # ------------------------------------------------------------------
    # Footer / boilerplate removal
    # ------------------------------------------------------------------

    # Pure boilerplate lines (unsubscribe links, legal links, help links, etc.)
    # removed whole-line. Handles optional leading bullet markers.
    _FOOTER_LINES_PATTERN = re.compile(
        r"^[ \t•·\-–—*|\u2022]*(?:"
        r"view(?: this e-?mail)? in(?: a| your)? browser|"
        r"manage preferences?|"
        r"unsubscribe|"
        r"update (?:your\s+)?e-?mail preferences|"
        r"privacy(?: and| &)? cookie policy|"
        r"privacy policies?|"
        r"terms(?: and| &| of)? (?:conditions?|service)|"
        r"help (?:pages?|center)|"
        r"you(?:'re| are) receiving this e-?mail|"
        r"this (?:e-?mail|message) was sent to|"
        r"please consider the environment|"
        r"can'?t see the(?: full| )? e-?mail|"
        r"not interested|"
        r"sent to\b|"
        r"read more\b|"
        r"subscription details"
        r")[ \t'\".,;:!?•·\u2022]*$",
        re.IGNORECASE | re.MULTILINE,
    )

    _COPYRIGHT_PATTERN = re.compile(
        r"(?:©|copyright\s*[©(c)]?)\s*(?:2\d{3})?[^\n]*",
        re.IGNORECASE,
    )

    # Decorative ASCII separators (---, ***, ===) — distinct from RFC 3676 "-- "
    _DECORATIVE_SEPARATOR_PATTERN = re.compile(
        r"^[ \t]*[-*_=]{3,}\s*$", re.MULTILINE
    )

    # Inline phrases that appear mid-line in boilerplate blocks (not whole lines).
    _INLINE_FOOTER_PATTERNS: List[Pattern[str]] = [
        re.compile(r"view(?: this e-?mail)? in(?: a| your)? browser", re.IGNORECASE),
    ]

    # ------------------------------------------------------------------
    # Important-link extraction (must run on raw HTML, before tags are
    # stripped — after that point the href is gone and only anchor text
    # survives, at which point it's indistinguishable from prose).
    # ------------------------------------------------------------------

    _ANCHOR_TAG_PATTERN = re.compile(
        r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )

    # Anchor text that signals a link worth keeping (event/registration/
    # purchase CTAs), matched against the anchor's inner text.
    _IMPORTANT_LINK_TEXT_PATTERN = re.compile(
        r"\b(register|rsvp|sign\s?up|shop\s?now|redeem|claim|book\s?now|"
        r"order\s?now|download|get\s?started|join\s?now|confirm|"
        r"view\s+(?:event|invite|ticket)|add\s+to\s+calendar)\b",
        re.IGNORECASE,
    )

    # Anchor inner text that is just a bare pointer word ("now", "here",
    # "this link") — only meaningful together with a CTA verb sitting in
    # the plain text immediately before the anchor, e.g. "RSVP <a>now</a>".
    _BARE_POINTER_TEXT_PATTERN = re.compile(
        r"^(now|here|this( link)?|link)$", re.IGNORECASE
    )
    _CTA_LEAD_IN_PATTERN = re.compile(
        r"\b(rsvp|register|sign\s?up|book|order|redeem|claim|confirm|join)\b",
        re.IGNORECASE,
    )
    # How many characters of plain text before an anchor to check for a
    # lead-in CTA verb. Wide enough to cover "Redeem the voucher <a>now</a>"
    # (verb + a few words of filler before the pointer word), not just an
    # immediately adjacent "RSVP <a>now</a>".
    _LEAD_IN_WINDOW = 40

    # Anchor text/href that is noise even if it matches a CTA-like word
    # above (e.g. "unsubscribe" is technically an imperative CTA too).
    _NOISE_LINK_TEXT_PATTERN = re.compile(
        r"\b(unsubscribe|manage preferences|privacy|terms|view in browser|"
        r"update.*preferences|help center)\b",
        re.IGNORECASE,
    )

    # Repeating "community digest" boilerplate blocks (forum-style
    # notification lists), e.g. "Stay engaged with the latest
    # discussions: <name> has posted the Discussion <title> (\n<time> ago"
    # Removed as a block regardless of what language names/titles are in.
    _DIGEST_BLOCK_PATTERN = re.compile(
        r"Stay engaged with the latest discussions:.*?(?=\n\s*\n\S|\Z)",
        re.DOTALL | re.IGNORECASE,
    )

    def clean(self, email: Email) -> Email:
        """
        Return a new `Email` with a cleaned body; all other fields preserved.

        Important links (registration/RSVP/purchase CTAs) found along the
        way are recorded on `self.last_extracted_links` as
        `{"url": ..., "text": ...}` dicts — reset at the start of every
        `clean()` call. The Email model has no field for these today, so
        they're exposed as a side channel rather than silently dropped;
        the orchestration layer can attach them to chunk metadata once
        that stage exists. Ask if you'd like `EmailModel.py` updated to
        carry them natively instead.
        """
        body = email.body or ""

        # Must run before HTML is stripped: after that point an <a href=...>
        # link's URL is gone and only its inner text remains, at which point
        # "important" and "tracking" links are indistinguishable.
        self.last_extracted_links: List[dict] = []
        body = self._extract_important_links(body)

        body = self._remove_html(body)
        body = self._normalize_unicode(body)
        body = self._remove_invisible_characters(body)
        body = self._remove_digest_blocks(body)
        body = self._remove_urls(body)
        body = self._remove_bracketed_labels(body)
        body = self._remove_reply_history(body)
        body = self._remove_signature(body)
        body = self._remove_decorative_separators(body)
        body = self._remove_inline_footers(body)
        body = self._remove_footers(body)
        body = self._remove_empty_parens(body)
        body = self._normalize_whitespace(body)
        body = self._strip_empty_lines(body)

        return email.model_copy(update={"body": body})

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_important_links(self, body: str) -> str:
        """
        Pull CTA-worthy links (register/RSVP/shop now/etc.) out of raw HTML
        anchor tags before HTML stripping runs. The matched anchor is
        replaced by its own inner text only (no markup, no href) so the
        surrounding sentence still reads naturally — e.g. "RSVP now" stays
        as plain readable text instead of being mangled or deleted. The
        href + text pair is appended to `self.last_extracted_links`.

        Anchors matching `_NOISE_LINK_TEXT_PATTERN` (unsubscribe, manage
        preferences, etc.) are never treated as important even if they also
        happen to match a CTA-style verb.
        """
        if not body or "<a" not in body.lower():
            return body

        def _replace(match: "re.Match[str]") -> str:
            href, inner_html = match.group(1), match.group(2)
            inner_text = re.sub(r"<[^>]+>", "", inner_html).strip()

            is_noise = bool(self._NOISE_LINK_TEXT_PATTERN.search(inner_text))
            is_important = bool(self._IMPORTANT_LINK_TEXT_PATTERN.search(inner_text))

            # Anchor text alone is a bare pointer ("now", "here") — check
            # the plain text immediately preceding the anchor for a CTA verb.
            if not is_important and self._BARE_POINTER_TEXT_PATTERN.match(inner_text):
                lead_in = body[max(0, match.start() - self._LEAD_IN_WINDOW):match.start()]
                lead_in_text = re.sub(r"<[^>]+>", "", lead_in)
                is_important = bool(self._CTA_LEAD_IN_PATTERN.search(lead_in_text))

            if is_important and not is_noise:
                self.last_extracted_links.append({"url": href, "text": inner_text})
                return f" {inner_text} "

            # Not an important link — leave the anchor tag in place so the
            # existing HTML→text→remove_urls path handles it as before.
            return match.group(0)

        return self._ANCHOR_TAG_PATTERN.sub(_replace, body)

    def _normalize_unicode(self, body: str) -> str:
        """
        NFKC-normalize the body so decorative/styled Unicode variants
        (e.g. mathematical bold "𝐑𝐞𝐰𝐚𝐫𝐝") fold back to plain ASCII and
        become both searchable and embeddable like ordinary text. This is
        script-agnostic — it does not touch legitimate non-Latin content
        (Arabic, Chinese, etc.), only compatibility/styling variants.
        """
        if not body:
            return ""
        return unicodedata.normalize("NFKC", body)

    def _remove_digest_blocks(self, body: str) -> str:
        """
        Remove repeating "community digest" boilerplate (forum-notification
        lists like "Stay engaged with the latest discussions: <name> has
        posted the Discussion <title> (\\n<relative time> ago"). Matched
        structurally, not by language, so it also removes instances whose
        names/titles happen to be in Arabic, Chinese, etc. — those aren't
        noise because of their script, they're noise because they're an
        unrelated forum digest bundled into the email.
        """
        if not body:
            return ""
        return self._DIGEST_BLOCK_PATTERN.sub(" ", body)

    def _remove_html(self, body: str) -> str:
        """Strip HTML tags/scripts/styles/blockquotes and decode entities."""
        if not body:
            return ""
        if looks_like_html(body):
            return html_to_text(body)
        return body

    def _remove_invisible_characters(self, body: str) -> str:
        """Replace invisible/spam-padding Unicode characters with spaces."""
        return remove_invisible_characters(body)

    def _remove_urls(self, body: str) -> str:
        """Strip URL strings (tracking, unsubscribe, legal links) keeping label text."""
        return remove_urls(body)

    def _remove_bracketed_labels(self, body: str) -> str:
        """Drop image/placeholder labels such as '[Coursera Logo]'."""
        return remove_bracketed_labels(body)

    def _remove_empty_parens(self, body: str) -> str:
        """Clean up empty parentheses left behind after URL removal."""
        return remove_empty_parens(body)

    def _remove_decorative_separators(self, body: str) -> str:
        """Remove decorative runs of dashes/asterisks/underscores used as separators."""
        if not body:
            return ""
        return self._DECORATIVE_SEPARATOR_PATTERN.sub(" ", body)

    def _remove_inline_footers(self, body: str) -> str:
        """Remove boilerplate phrases that appear inside otherwise-content lines."""
        if not body:
            return ""
        for pattern in self._INLINE_FOOTER_PATTERNS:
            body = pattern.sub(" ", body)
        return body

    def _remove_footers(self, body: str) -> str:
        """Remove whole-line footer/legal/unsubscribe boilerplate and copyright text."""
        if not body:
            return ""
        body = self._FOOTER_LINES_PATTERN.sub(" ", body)
        body = self._COPYRIGHT_PATTERN.sub(" ", body)
        return body

    def _remove_reply_history(self, body: str) -> str:
        """
        Truncate the body at the earliest quoted/forwarded-reply marker,
        then drop any remaining lines that are pure '>' quote markers.
        """
        if not body:
            return ""

        earliest_index: Optional[int] = None
        for pattern in self._REPLY_HISTORY_PATTERNS:
            match = pattern.search(body)
            if match and (earliest_index is None or match.start() < earliest_index):
                earliest_index = match.start()

        if earliest_index is not None:
            body = body[:earliest_index]

        return self._QUOTE_LINE_PATTERN.sub(" ", body)

    def _remove_signature(self, body: str) -> str:
        """Truncate the body at the first sign-off phrase or RFC 3676 '-- ' delimiter."""
        if not body:
            return ""

        cut_index: Optional[int] = None

        sig_delim_match = self._SIG_DELIMITER_PATTERN.search(body)
        if sig_delim_match:
            cut_index = sig_delim_match.start()

        signoff_match = self._SIGNOFF_PATTERN.search(body)
        if signoff_match and (cut_index is None or signoff_match.start() < cut_index):
            cut_index = signoff_match.start()

        if cut_index is not None:
            body = body[:cut_index]

        return body

    def _normalize_whitespace(self, body: str) -> str:
        """Coll apse runs of spaces/tabs and trim trailing whitespace per line."""
        return normalize_whitespace(body)

    def _strip_empty_lines(self, body: str) -> str:
        """Collapse consecutive blank lines and trim leading/trailing blank space."""
        return strip_empty_lines(body)