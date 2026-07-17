# Component: Takeaway
import streamlit as st


def show_takeaway(
    text: str,
    title: str = "Key Takeaway",
) -> None:
    """Display the central conclusion at the end of a page."""

    st.divider()

    st.success(
        f"""
        **✅ {title}**

        {text}
        """
    )