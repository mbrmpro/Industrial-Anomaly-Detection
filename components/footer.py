import streamlit as st

from utils.config import VERSION

def show_footer():

    st.divider()

    st.caption(
        f"Industrial Anomaly Detection • Version {VERSION}"
    )