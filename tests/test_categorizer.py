import json

import pandas as pd
import pytest

from categorizer import (
    INCOME_CATEGORY,
    _apply_rule_based,
    categorize_batch,
    normalize_category,
)
from tests.conftest import transactions_in_prompt


def make_df(rows):
    df = pd.DataFrame(rows, columns=["description", "merchant", "amount", "direction"])
    df["original_category"] = ""
    df["category"] = ""
    return df


@pytest.mark.parametrize("merchant, expected", [
    ("Coca-Cola Store", ""),           # "ola" must not match inside a word
    ("Motorola Service", ""),
    ("Ola Cabs", "Transport"),
    ("Amazon Prime", "Subscriptions"),  # longest keyword wins over "amazon"
    ("Amazon", "Shopping"),
    ("Zomato Dining", "Entertainment"),
    ("SWIGGY", "Food & Dining"),
])
def test_rules_use_whole_words_and_longest_match(merchant, expected):
    df = pd.DataFrame({"merchant": [merchant], "category": [""]})
    assert _apply_rule_based(df).loc[0, "category"] == expected


def test_rules_fall_back_to_description():
    df = pd.DataFrame({"merchant": ["UPI-8712"], "description": ["UPI/P2M/Swiggy/A/350"], "category": [""]})
    assert _apply_rule_based(df).loc[0, "category"] == "Food & Dining"


def test_normalize_category():
    assert normalize_category("food & dining") == "Food & Dining"
    assert normalize_category(" Transport ") == "Transport"
    assert normalize_category("DR") is None
    assert normalize_category(None) is None


def test_only_valid_original_categories_are_trusted():
    df = make_df([("a", "x", 10, "debit"), ("b", "y", 10, "debit")])
    df["original_category"] = ["Groceries", "UPI"]
    out = categorize_batch(df, client=None, use_cache=False)
    assert out.loc[0, "category"] == "Groceries"
    assert out.loc[0, "category_source"] == "existing"
    assert out.loc[1, "category"] == "Other"
    assert out.loc[1, "category_source"] == "fallback"


def test_credits_become_income_without_llm(fake_client):
    client = fake_client(lambda kwargs: pytest.fail("LLM should not be called"))
    df = make_df([("SALARY", "ACME", 50000, "credit")])
    out = categorize_batch(df, client, use_cache=False)
    assert out.loc[0, "category"] == INCOME_CATEGORY
    assert client.calls == []


def _responder(kwargs):
    labels = {"Corner Bistro": "food & dining", "Mystery Ltd": "Gibberish"}
    txns = transactions_in_prompt(kwargs)
    return json.dumps({"categories": [{"index": t["index"], "category": labels.get(t["merchant"], "Other")} for t in txns]})


def test_llm_results_are_normalized_and_deduplicated(fake_client, tmp_path):
    client = fake_client(_responder)
    df = make_df([
        ("Corner Bistro dinner", "Corner Bistro", 900, "debit"),
        ("Corner Bistro dinner", "Corner Bistro", 950, "debit"),   # same text -> one LLM slot
        ("Mystery charge", "Mystery Ltd", 120, "debit"),
    ])
    out = categorize_batch(df, client, cache_path=str(tmp_path / "cache.json"))
    assert list(out["category"]) == ["Food & Dining", "Food & Dining", "Other"]
    assert list(out["category_source"]) == ["llm", "llm", "llm"]
    assert len(client.calls) == 1
    assert len(transactions_in_prompt(client.calls[0])) == 2


def test_cache_is_reused_without_junk_rows(fake_client, tmp_path):
    cache = str(tmp_path / "cache.json")
    df = make_df([("Corner Bistro dinner", "Corner Bistro", 900, "debit")])
    categorize_batch(df, fake_client(_responder), cache_path=cache)

    second_client = fake_client(lambda kwargs: pytest.fail("cache hit expected"))
    out = categorize_batch(df, second_client, cache_path=cache)
    assert len(out) == 1
    assert out.loc[0, "category"] == "Food & Dining"
    assert out.loc[0, "category_source"] == "cache"


def test_corrupt_cache_is_ignored(fake_client, tmp_path):
    cache = tmp_path / "cache.json"
    cache.write_text('{"abc": {', encoding="utf-8")
    df = make_df([("Corner Bistro dinner", "Corner Bistro", 900, "debit")])
    out = categorize_batch(df, fake_client(_responder), cache_path=str(cache))
    assert out.loc[0, "category"] == "Food & Dining"
    assert json.loads(cache.read_text(encoding="utf-8"))  # rewritten as valid JSON


def test_progress_never_exceeds_total(fake_client, tmp_path, monkeypatch):
    monkeypatch.setattr("categorizer.BATCH_SIZE", 2)
    df = make_df([(f"Shop {i}", f"Shop {i}", 10, "debit") for i in range(5)])
    seen = []
    categorize_batch(df, fake_client(_responder), cache_path=str(tmp_path / "c.json"),
                     progress_callback=lambda done, total: seen.append((done, total)))
    assert seen and all(done <= total for done, total in seen)
    assert seen[-1] == (5, 5)


def test_failed_llm_falls_back_to_other_and_is_not_cached(fake_client, tmp_path):
    cache = tmp_path / "cache.json"
    client = fake_client(lambda kwargs: RuntimeError("rate limited"))
    df = make_df([("Corner Bistro dinner", "Corner Bistro", 900, "debit")])
    out = categorize_batch(df, client, cache_path=str(cache))
    assert out.loc[0, "category"] == "Other"
    assert out.loc[0, "category_source"] == "fallback"
    assert json.loads(cache.read_text(encoding="utf-8")) == {}


def test_pii_is_masked_before_sending(fake_client, tmp_path):
    client = fake_client(_responder)
    df = make_df([("UPI/rahul.sharma@okhdfcbank/9876543210", "rahul.sharma@okhdfcbank", 500, "debit")])
    categorize_batch(df, client, cache_path=str(tmp_path / "c.json"))
    sent = json.dumps(transactions_in_prompt(client.calls[0]))
    assert "rahul.sharma" not in sent
    assert "9876543210" not in sent
    assert "<UPI_ID>" in sent


def test_input_frame_is_not_mutated():
    df = make_df([("Swiggy order", "Swiggy", 300, "debit")])
    categorize_batch(df, client=None, use_cache=False)
    assert df.loc[0, "category"] == ""
    assert "category_source" not in df.columns
