"""Private mobile-only entrypoint.

Run: streamlit run mobile_app.py
IMPORTANT: st.navigation intentionally prevents the other pages/ from being
exposed when deploying the same repository to Streamlit Community Cloud.
"""
import streamlit as st

st.set_page_config(
    page_title="Valuation Lite",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Streamlit ignores automatic pages/ discovery when st.navigation is used.
page = st.navigation(
    [st.Page("mobile_valuation.py", title="Stock Valuation", icon="📈", default=True)],
    position="hidden",
)
page.run()
