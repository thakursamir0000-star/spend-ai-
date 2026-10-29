import json
from types import SimpleNamespace

import pytest

import llm_client
from llm_client import candidate_models, chat, chat_json
from privacy import mask_pii


def test_failover_moves_to_next_model(fake_client):
    def responder(kwargs):
        if kwargs["model"] == "openai/gpt-oss-20b":
            return RuntimeError("model decommissioned")
        return "ok"

    client = fake_client(responder)
    assert chat(client, "hi") == "ok"
    assert [c["model"] for c in client.calls][:2] == ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]


class MissingModel(Exception):
    status_code = 404


def test_missing_model_is_skipped_after_first_404(fake_client):
    def responder(kwargs):
        if kwargs["model"] == "openai/gpt-oss-20b":
            return MissingModel("The model does not exist (model_not_found)")
        return "ok"

    client = fake_client(responder)
    assert chat(client, "first") == "ok"
    assert chat(client, "second") == "ok"
    models = [c["model"] for c in client.calls]
    assert models.count("openai/gpt-oss-20b") == 1
    assert "openai/gpt-oss-20b" not in candidate_models()


def test_other_errors_are_retried_next_time(fake_client):
    client = fake_client(lambda kwargs: RuntimeError("rate limited"))
    chat(client, "first")
    assert "openai/gpt-oss-20b" in candidate_models()


class RateLimited(Exception):
    status_code = 429

    def __init__(self, retry_after):
        super().__init__("Rate limit reached")
        self.response = SimpleNamespace(headers={"retry-after": str(retry_after)})


def test_default_chain_has_no_retired_models():
    assert candidate_models() == ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]


def test_daily_rate_limit_skips_model_until_it_resets(fake_client, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(llm_client.time, "monotonic", lambda: now[0])

    def responder(kwargs):
        if kwargs["model"] == "openai/gpt-oss-20b":
            return RateLimited(417)
        return "ok"

    client = fake_client(responder)
    assert chat(client, "first") == "ok"
    assert chat(client, "second") == "ok"
    assert [c["model"] for c in client.calls].count("openai/gpt-oss-20b") == 1
    assert "openai/gpt-oss-20b" not in candidate_models()

    now[0] += 418
    assert candidate_models()[0] == "openai/gpt-oss-20b"


def test_short_rate_limit_is_not_remembered(fake_client):
    # The SDK has already waited this one out; the next call should try the model again.
    client = fake_client(lambda kwargs: RateLimited(15) if kwargs["model"] == "openai/gpt-oss-20b" else "ok")
    assert chat(client, "hi") == "ok"
    assert "openai/gpt-oss-20b" in candidate_models()


def test_low_reasoning_effort_only_for_gpt_oss(fake_client, monkeypatch):
    monkeypatch.setenv("GROQ_FALLBACK_MODELS", "qwen/qwen3.8-27b")
    client = fake_client(lambda kwargs: RuntimeError("down") if "gpt-oss" in kwargs["model"] else "{}")
    chat_json(client, "hi")
    gpt_oss, qwen = client.calls
    assert gpt_oss["reasoning_effort"] == "low"
    assert "reasoning_effort" not in qwen
    assert qwen["response_format"] == {"type": "json_object"}


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
