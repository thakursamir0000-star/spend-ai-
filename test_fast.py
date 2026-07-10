"""Test categorization speed with sample data."""
import sys, types, pandas as pd

mock_st = types.ModuleType('streamlit')
mock_st.info = lambda x: print(f'INFO: {x}')
mock_st.success = lambda x: print(f'SUCCESS: {x}')
mock_st.error = lambda x: print(f'ERROR: {x}')
mock_st.warning = lambda x: print(f'WARN: {x}')
mock_st.session_state = types.SimpleNamespace(column_mapping=None)
sys.modules['streamlit'] = mock_st

from data_cleaner import clean_dataframe
from categorizer import categorize_batch

df = pd.read_csv('sample_transactions.csv')
cleaned = clean_dataframe(df)

import time
start = time.time()
result = categorize_batch(cleaned, client=None)
elapsed = time.time() - start

categorized = (result['category'] != '').sum()
uncategorized = (result['category'] == '').sum()
print(f'Categorized: {categorized}/{len(result)} in {elapsed:.2f}s')
print(f'Uncategorized: {uncategorized}')
if uncategorized == 0:
    print('PASS: All transactions categorized without any LLM calls')
else:
    print(f'FAIL: {uncategorized} still uncategorized')
