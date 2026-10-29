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
