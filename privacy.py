"""Mask personal identifiers before transaction text is sent to a third-party LLM.

Bank and UPI descriptions often contain UPI IDs, phone numbers, emails, and
account/card/reference numbers. None of these help categorization, so they are
replaced with placeholders. Set MASK_PII=false to disable (e.g. for debugging).
"""
import os
import re

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_UPI_ID = re.compile(r"\b[\w.-]{2,}@[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")
# Card numbers, account numbers, UTR/reference numbers: 9+ digits, optionally grouped.
_LONG_NUMBER = re.compile(r"(?<!\d)\d(?:[\s-]?\d){8,}(?!\d)")
_MASKED_CARD = re.compile(r"\b[Xx*]{4,}\d{2,6}\b")

_PATTERNS = [
    (_EMAIL, "<EMAIL>"),
    (_UPI_ID, "<UPI_ID>"),
    (_PHONE, "<PHONE>"),
    (_MASKED_CARD, "<CARD>"),
    (_LONG_NUMBER, "<NUMBER>"),
]


def masking_enabled():
    return os.environ.get("MASK_PII", "true").strip().lower() not in {"0", "false", "no"}


def mask_pii(text):
    """Return `text` with personal identifiers replaced by placeholders."""
    if not isinstance(text, str) or not text or not masking_enabled():
        return text
    for pattern, placeholder in _PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


_TOKEN = re.compile(r"<([A-Z]+(?:_[A-Z]+)*)_(\d+)>")


class Pseudonymizer:
    """Replace identifiers with numbered placeholders and remember the originals.

    Unlike mask_pii, two different UPI IDs become <UPI_ID_1> and <UPI_ID_2>, so
    they stay distinct (grouped results don't collide), the same value always
    gets the same placeholder, and unmask() can put the real values back locally
    after the LLM has answered. Use one instance per question.
    """

    def __init__(self):
        self._token_for = {}
        self._value_for = {}
        self._counts = {}

    def _token(self, value, placeholder):
        if value not in self._token_for:
            kind = placeholder.strip("<>")
            self._counts[kind] = self._counts.get(kind, 0) + 1
            token = f"<{kind}_{self._counts[kind]}>"
            self._token_for[value] = token
            self._value_for[token] = value
        return self._token_for[value]

    def mask(self, text):
        if not isinstance(text, str) or not text or not masking_enabled():
            return text
        for pattern, placeholder in _PATTERNS:
            text = pattern.sub(lambda m, p=placeholder: self._token(m.group(0), p), text)
        return text

    def unmask(self, text):
        if not isinstance(text, str) or not self._value_for:
            return text
        return _TOKEN.sub(lambda m: self._value_for.get(m.group(0), m.group(0)), text)
