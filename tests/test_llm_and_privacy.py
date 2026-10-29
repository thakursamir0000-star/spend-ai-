import json

import pytest

from llm_client import candidate_models, chat, chat_json
from privacy import mask_pii


def test_failover_moves_to_next_model(fake_client):
    def responder(kwargs):
        if kwargs["model"] == "openai/gpt-oss-20b":
            return RuntimeError("model decommissioned")
        return "ok"

    client = fake_client(responder)
    assert chat(client, "hi") == "ok"
    assert [c["model"] for c in client.calls][:2] == ["openai/gpt-oss-20b", "qwen/qwen3.6-27b"]


def test_all_models_failing_returns_none(fake_client):
    client = fake_client(lambda kwargs: RuntimeError("down"))
    assert chat(client, "hi") is None
    assert chat_json(client, "hi") is None
    assert chat(None, "hi") is None


def test_invalid_json_tries_next_model(fake_client):
    answers = iter(["not json", json.dumps({"a": 1})])
    client = fake_client(lambda kwargs: next(answers))
    assert chat_json(client, "hi") == {"a": 1}
    assert len(client.calls) == 2


def test_model_list_is_configurable(monkeypatch):
    monkeypatch.setenv("GROQ_QA_MODEL", "my/model")
    monkeypatch.setenv("GROQ_FALLBACK_MODELS", "a/one, my/model ,b/two")
    assert candidate_models("GROQ_QA_MODEL") == ["my/model", "a/one", "b/two"]


@pytest.mark.parametrize("text, hidden, placeholder", [
    ("UPI/rahul.s@okicici/Payment", "rahul.s@okicici", "<UPI_ID>"),
    ("paid 9876543210 via UPI", "9876543210", "<PHONE>"),
    ("call +91 98765 43210", None, None),
    ("NEFT to a/c 123456789012", "123456789012", "<NUMBER>"),
    ("card 4111 1111 1111 1111", "4111 1111 1111 1111", "<NUMBER>"),
    ("card XXXX1234", "XXXX1234", "<CARD>"),
    ("mail me at a.b@example.com", "a.b@example.com", "<EMAIL>"),
])
def test_mask_pii(text, hidden, placeholder):
    masked = mask_pii(text)
    if hidden:
        assert hidden not in masked
        assert placeholder in masked


def test_mask_keeps_merchant_words():
    assert mask_pii("Swiggy order 01 Jan") == "Swiggy order 01 Jan"
    assert mask_pii("Amount 450.00") == "Amount 450.00"


def test_masking_can_be_disabled(monkeypatch):
    monkeypatch.setenv("MASK_PII", "false")
    assert mask_pii("9876543210") == "9876543210"
