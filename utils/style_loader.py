from pathlib import Path
import streamlit as st


def load_css():
    """Load custom CSS stylesheet."""

    css_file = Path(__file__).parent.parent / "assets" / "css" / "style.css"

    with open(css_file, encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True,
        )