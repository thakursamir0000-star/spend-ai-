import math
from pathlib import Path

import pandas as pd
import pytest

from categorizer import categorize_batch
from data_cleaner import clean_dataframe, spend_only
from evaluate import QA_CASES, _matches

SAMPLE = Path(__file__).resolve().parents[1] / "sample_transactions.csv"


@pytest.fixture(scope="module")
def sample_spend():
    df = categorize_batch(clean_dataframe(pd.read_csv(SAMPLE)), None, use_cache=False)
    return spend_only(df)


def test_every_ground_truth_is_a_real_answer(sample_spend):
    for question, truth in QA_CASES:
        expected = truth(sample_spend)
        if isinstance(expected, dict):
            assert expected, question
        else:
            assert math.isfinite(float(expected)) and float(expected) > 0, question


def test_last_month_is_relative_to_the_data(sample_spend):
    assert sample_spend["date"].max() == pd.Timestamp("2026-06-30")
    truth = dict(QA_CASES)["How much did I spend last month?"]
    may = sample_spend.loc[sample_spend["month"] == "2026-05", "amount"].sum()
    assert truth(sample_spend) == may


def test_matching_tolerance():
    assert _matches(100.4, 100.0)
    assert not _matches(110.0, 100.0)
    assert _matches({"2026-01": 10.0}, {"2026-01": 10.2})
    assert not _matches(None, 1.0)
