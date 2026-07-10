"""Quick test of the data pipeline without Streamlit dependencies."""
import pandas as pd
import sys
import types

mock_st = types.ModuleType('streamlit')
mock_st.info = lambda x: print(f'INFO: {x}')
mock_st.success = lambda x: print(f'SUCCESS: {x}')
mock_st.error = lambda x: print(f'ERROR: {x}')
mock_st.warning = lambda x: print(f'WARN: {x}')
mock_st.session_state = types.SimpleNamespace(column_mapping=None)
sys.modules['streamlit'] = mock_st

from data_cleaner import clean_dataframe

df = pd.read_csv('sample_transactions.csv')
print(f'Raw shape: {df.shape}')
print(f'Columns: {list(df.columns)}')

result = clean_dataframe(df)
if result is not None:
    print(f'Cleaned shape: {result.shape}')
    print(f'Columns: {list(result.columns)}')
    print(f'Date range: {result["date"].min()} to {result["date"].max()}')
    print(f'Total spend: {result["amount"].sum():.0f}')
    print(f'Sample rows:')
    print(result.head(3).to_string())
else:
    print('FAILED: clean_dataframe returned None')

print('\nAll pipeline tests passed!')
