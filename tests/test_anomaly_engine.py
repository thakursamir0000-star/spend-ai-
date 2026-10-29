import json

import pandas as pd

from anomaly_engine import detect_anomalies, find_anomalies


def frame(rows):
    df = pd.DataFrame(rows, columns=["date", "amount", "category", "merchant"])
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["direction"] = "debit"
    return df


def history():
    """Jan–Apr 2026: steady ₹1,000/month on food at the same restaurant."""
    return [(f"2026-0{m}-10", 1000.0, "Food & Dining", "Corner Bistro") for m in range(1, 5)]


def test_spike_is_measured_against_the_data_not_the_clock():
    df = frame(history() + [("2026-05-10", 3000.0, "Food & Dining", "Corner Bistro")])
    spikes = [a for a in find_anomalies(df) if a["type"] == "category_spike"]
    assert len(spikes) == 1
    assert spikes[0]["month"] == "2026-05"
    assert spikes[0]["average"] == 1000.0
    assert spikes[0]["deviation_pct"] == 200.0


def test_small_spikes_are_ignored():
    df = frame(history() + [("2026-05-10", 1400.0, "Food & Dining", "Corner Bistro")])
    assert find_anomalies(df) == []


def test_new_recurring_merchant_uses_latest_date_in_data():
    rows = history() + [
        ("2026-05-02", 999.0, "Subscriptions", "NewStream"),
        ("2026-06-02", 999.0, "Subscriptions", "NewStream"),
        ("2026-06-03", 1000.0, "Food & Dining", "Corner Bistro"),
    ]
    found = [a for a in find_anomalies(frame(rows)) if a["type"] == "new_recurring"]
    assert found == [{"type": "new_recurring", "merchant": "NewStream", "count": 2, "total": 1998.0}]


def test_short_history_does_not_flag_every_merchant():
    rows = [("2026-06-01", 100.0, "Shopping", "A"), ("2026-06-05", 100.0, "Shopping", "A")]
    assert find_anomalies(frame(rows)) == []


def test_credits_are_excluded():
    df = frame(history() + [("2026-05-10", 90000.0, "Food & Dining", "Corner Bistro")])
    df.loc[df.index[-1], "direction"] = "credit"
    assert find_anomalies(df) == []


def test_fallback_messages_without_llm():
    df = frame(history() + [("2026-05-10", 3000.0, "Food & Dining", "Corner Bistro")])
    messages = detect_anomalies(df, client=None)
    assert messages == [{
        "type": "category_spike",
        "message": "You spent ₹3,000 on Food & Dining in 2026-05 (+200% vs your average of ₹1,000)",
    }]


def test_llm_wording_is_used_but_types_are_ours(fake_client):
    df = frame(history() + [("2026-05-10", 3000.0, "Food & Dining", "Corner Bistro")])
    client = fake_client(lambda kwargs: json.dumps({"anomalies": [{"type": "whatever", "message": "Food tripled!"}]}))
    assert detect_anomalies(df, client) == [{"type": "category_spike", "message": "Food tripled!"}]


def test_malformed_llm_output_falls_back(fake_client):
    df = frame(history() + [("2026-05-10", 3000.0, "Food & Dining", "Corner Bistro")])
    client = fake_client(lambda kwargs: json.dumps({"anomalies": []}))
    assert detect_anomalies(df, client)[0]["message"].startswith("You spent ₹3,000")
