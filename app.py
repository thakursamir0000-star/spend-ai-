import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from groq import Groq

from data_cleaner import clean_dataframe
from categorizer import categorize_batch, CATEGORIES
from anomaly_engine import detect_anomalies
from query_engine import answer_question

st.set_page_config(
    page_title="AI Spend Insights Bot",
    page_icon="💰",
    layout="wide"
)

st.title("💰 AI Spend Insights Bot")
st.markdown("Upload your bank/UPI transaction CSV to get auto-categorized insights, anomaly detection, and natural-language Q&A.")

if 'df' not in st.session_state:
    st.session_state.df = None
if 'categorized' not in st.session_state:
    st.session_state.categorized = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'anomalies' not in st.session_state:
    st.session_state.anomalies = []
if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {}


def get_client():
    api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        return None
    return Groq(api_key=api_key)


client = get_client()

if not client:
    st.warning("GROQ_API_KEY not found. Set it in environment variables or .streamlit/secrets.toml. Categorization and Q&A will be limited.")

SAMPLE_CSV_PATH = "sample_transactions.csv"

with st.sidebar:
    st.header("1. Upload Data")
    uploaded = st.file_uploader("Choose a transaction CSV", type=["csv"], key="csv_uploader")

    if uploaded:
        raw_df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(raw_df)} rows from {uploaded.name}")
    elif os.path.exists(SAMPLE_CSV_PATH):
        st.info("No file uploaded. Using sample data.")
        raw_df = pd.read_csv(SAMPLE_CSV_PATH)
        st.success(f"Loaded {len(raw_df)} rows from sample data")
    else:
        raw_df = None
        st.info("Upload a CSV or place 'sample_transactions.csv' in the app directory.")

    st.header("2. Process Data")
    if raw_df is not None and st.button("Clean & Categorize", type="primary"):
        with st.spinner("Cleaning data…"):
            df = clean_dataframe(raw_df)
        if df is not None:
            progress_bar = st.progress(0, text="Categorizing transactions…")
            status_text = st.empty()
            def on_progress(current, total):
                progress_bar.progress(current / total, text=f"Categorized {current}/{total} transactions")

            if client:
                df = categorize_batch(df, client, progress_callback=on_progress)
            else:
                st.error("Groq client not configured. Set GROQ_API_KEY.")

            progress_bar.empty()
            status_text.empty()
            st.session_state.df = df
            st.session_state.categorized = True
            st.session_state.messages = []
            st.rerun()

    st.header("3. About")
    st.markdown("""
    **AI Spend Insights Bot** analyzes your transaction data using LLMs.
    
    - Auto-categorizes expenses (Groq Cloud)
    - Detects spending anomalies
    - Answers natural-language questions
    """)

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Anomalies", "💬 Ask AI"])

with tab1:
    df = st.session_state.df
    if df is not None and not df.empty:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Category Breakdown")
            cat_totals = df.groupby('category')['amount'].sum().sort_values(ascending=False)
            fig = px.pie(
                values=cat_totals.values,
                names=cat_totals.index,
                title="Spend by Category",
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Top 5 Merchants")
            top_merchants = df.groupby('merchant')['amount'].sum().sort_values(ascending=False).head(5)
            fig2 = px.bar(
                x=top_merchants.values,
                y=top_merchants.index,
                orientation='h',
                title="Top Merchants by Spend",
                labels={'x': 'Total Spend', 'y': ''},
                text_auto='.0f'
            )
            fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            st.subheader("Monthly Trend")
            monthly = df.groupby('month')['amount'].sum().reset_index()
            monthly.columns = ['month', 'amount']
            fig3 = px.line(
                monthly, x='month', y='amount',
                title="Monthly Spend",
                markers=True,
                labels={'amount': 'Total Spend', 'month': 'Month'}
            )
            st.plotly_chart(fig3, use_container_width=True)

            st.subheader("Category Monthly Breakdown")
            cat_monthly = df.groupby(['month', 'category'])['amount'].sum().reset_index()
            fig4 = px.area(
                cat_monthly, x='month', y='amount', color='category',
                title="Category Spend Over Time",
                labels={'amount': 'Spend', 'month': 'Month', 'category': 'Category'}
            )
            st.plotly_chart(fig4, use_container_width=True)

        st.subheader("Transaction Data")
        display_cols = ['date', 'description', 'merchant', 'amount', 'category', 'month']
        avail_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[avail_cols].sort_values('date', ascending=False).head(50), use_container_width=True)
    else:
        st.info("Upload a CSV and click 'Clean & Categorize' to see the dashboard.")

with tab2:
    df = st.session_state.df
    if df is not None and not df.empty:
        if st.button("Detect Anomalies", type="primary"):
            with st.spinner("Analyzing for anomalies…"):
                anomalies = detect_anomalies(df, client)
                st.session_state.anomalies = anomalies

        if st.session_state.anomalies:
            st.subheader(f"Found {len(st.session_state.anomalies)} Anomalies")
            for a in st.session_state.anomalies:
                msg = a.get("message", "")
                atype = a.get("type", "")
                if "spike" in atype:
                    st.error(f"📈 {msg}")
                elif "recurring" in atype:
                    st.warning(f"🔄 {msg}")
                else:
                    st.info(msg)
        else:
            st.info("Click 'Detect Anomalies' to scan for unusual spending patterns.")

        st.subheader("Monthly Category Comparison")
        months = sorted(df['month'].unique())
        if len(months) >= 2:
            comp_data = df[df['month'].isin(months[-3:])]
            cat_monthly = comp_data.groupby(['month', 'category'])['amount'].sum().reset_index()
            fig = px.bar(
                cat_monthly, x='month', y='amount', color='category',
                title="Monthly Category Comparison (Last 3 Months)",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload and process data first.")

with tab3:
    df = st.session_state.df
    if df is not None and not df.empty and client:
        st.subheader("Ask about your spending")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("e.g. How much did I spend on food last month?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing…"):
                    answer = answer_question(prompt, df, client)
                st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})

    elif df is not None and not df.empty and not client:
        st.warning("GROQ API key not configured. Set GROQ_API_KEY to enable the Q&A feature.")
    else:
        st.info("Upload and process data first.")
