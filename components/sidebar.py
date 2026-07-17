import streamlit as st


def show_sidebar():

    with st.sidebar:

        st.title("Industrial AI")

        st.divider()

        st.info(
            """
            Industrial Anomaly Detection

            MVTec AD Dataset
            """
        )

        st.divider()

        st.success("Sprint 1")