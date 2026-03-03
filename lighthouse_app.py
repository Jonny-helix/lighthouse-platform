"""
LIGHTHOUSE — Practice Intelligence Platform
Streamlit entry point.
"""
import streamlit as st

st.set_page_config(
    page_title="LIGHTHOUSE",
    page_icon="🔦",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = st.navigation([
    st.Page("pages/home.py",            title="Home",            icon="🏠"),
    st.Page("pages/practitioner_kb.py", title="Practitioner KB", icon="📚"),
    st.Page("pages/observatory.py",     title="Observatory",     icon="🔭"),
    st.Page("pages/settings.py",        title="Settings",        icon="⚙️"),
])
pages.run()
