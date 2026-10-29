import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeCompletions:
    """Stands in for `client.chat.completions`; `responder(kwargs)` returns text or raises."""

    def __init__(self, responder):
        self.responder = responder
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.responder(kwargs)
        if isinstance(content, Exception):
            raise content
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeClient:
    def __init__(self, responder):
        self.chat = SimpleNamespace(completions=FakeCompletions(responder))

    @property
    def calls(self):
        return self.chat.completions.calls


def prompt_of(kwargs):
    return kwargs["messages"][0]["content"]


def transactions_in_prompt(kwargs):
    """The JSON transaction list embedded at the end of the categorization prompt."""
    return json.loads(prompt_of(kwargs).split("Transactions:\n", 1)[1])


@pytest.fixture
def fake_client():
    return FakeClient


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("GROQ_FALLBACK_MODELS", "GROQ_CATEGORIZER_MODEL", "GROQ_QA_MODEL", "MASK_PII"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def txns():
    """Six months of small, hand-checkable spending plus one salary credit."""
    rows = [
        ("2026-01-05", 1000.0, "debit", "Shopping", "Amazon"),
        ("2026-01-20", 200.0, "debit", "Food & Dining", "Swiggy"),
        ("2026-02-10", 300.0, "debit", "Food & Dining", "Swiggy Instamart"),
        ("2026-03-15", 400.0, "debit", "Transport", "Uber"),
        ("2026-04-01", 50000.0, "credit", "Income", "ACME Corp"),
        ("2026-05-12", 250.0, "debit", "Food & Dining", "Zomato"),
        ("2026-06-10", 50.0, "debit", "Shopping", "Amazon"),
        ("2026-06-28", 700.0, "debit", "Transport", "Uber"),
    ]
    df = pd.DataFrame(rows, columns=["date", "amount", "direction", "category", "merchant"])
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["description"] = df["merchant"]
    return df
