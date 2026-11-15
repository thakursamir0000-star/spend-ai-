"""CSV import regression tests: real CSV bytes through the loader the app uses, and the app itself."""
import io
from pathlib import Path

import pytest

from categorizer import categorize_batch
from data_cleaner import CREDIT, DEBIT, clean_dataframe, read_transactions_csv, spend_only

ROOT = Path(__file__).resolve().parents[1]

SIGNED_CSV = b"Date,Description,Amount\n2026-06-01,Myntra refund,+1200\n2026-06-02,Lunch at cafe,-450\n"


def import_csv(data):
    """The app's import pipeline: read the uploaded bytes, clean, categorize (rules only)."""
    return categorize_batch(clean_dataframe(read_transactions_csv(io.BytesIO(data))), None, use_cache=False)


def totals(df):
    return spend_only(df)["amount"].sum(), df.loc[df["direction"] == CREDIT, "amount"].sum()


def test_plus_and_minus_amounts_keep_their_meaning():
    # pandas alone reads "+1200" as 1200, and a plain number counts as spending.
    df = import_csv(SIGNED_CSV)
    assert df.set_index("description")["direction"].to_dict() == {"Myntra refund": CREDIT, "Lunch at cafe": DEBIT}
    assert totals(df) == (450, 1200)


@pytest.mark.parametrize("csv_text, spend, received", [
    # plain positive amounts are spending; currency text and thousands separators
    ('Date,Description,Amount\n2026-01-01,Netflix,"₹ 649.00"\n2026-01-02,Rent,"12,000"\n', 12649, 0),
    # Cr / Dr suffixes
    ('Date,Description,Amount\n2026-01-01,Salary,"50,000.00 Cr"\n2026-01-03,Swiggy,450 Dr\n', 450, 50000),
    # separate withdrawal and deposit columns with empty cells
    ("Date,Narration,Withdrawal Amt.,Deposit Amt.\n2026-01-01,UPI Swiggy,450.00,\n2026-01-02,Refund,,1200.00\n",
     450, 1200),
    # a DR/CR type column next to an unsigned amount
    ("Date,Description,Type,Amount\n2026-01-01,Uber,DR,300\n2026-01-02,Cashback,CR,50\n", 300, 50),
    # brackets for money out
    ('Date,Description,Amount\n2026-01-01,Amazon,(999.50)\n', 999.5, 0),
])
def test_existing_csv_formats_still_import(csv_text, spend, received):
    assert totals(import_csv(csv_text.encode("utf-8"))) == pytest.approx((spend, received))


@pytest.mark.parametrize("csv_text, expected", [
    ("Date,Description,Amount\n03-02-2026,x,10\n13-02-2026,y,10\n", ["2026-02-03", "2026-02-13"]),  # day first
    ("Date,Description,Amount\n2026-02-03,x,10\n2026/02/13,y,10\n", ["2026-02-03", "2026-02-13"]),  # year first
])
def test_dates_are_read_as_before(csv_text, expected):
    df = import_csv(csv_text.encode("utf-8"))
    assert df["date"].dt.strftime("%Y-%m-%d").tolist() == expected


def test_app_dashboard_shows_spend_and_refund_from_a_signed_csv(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    for var in ("LLM_PROVIDER", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)  # rules only: no LLM calls from a test
    # The app falls back to ./sample_transactions.csv when nothing is uploaded.
    (tmp_path / "sample_transactions.csv").write_bytes(SIGNED_CSV)
    monkeypatch.chdir(tmp_path)

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    next(b for b in at.button if b.label == "Clean & Categorize").click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Total spend"] == "₹450"
    assert metrics["Income / credits"] == "₹1,200"
    assert metrics["Net"] == "₹750"
