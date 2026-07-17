import streamlit as st


def show_page_header(
    title: str,
    subtitle: str,
    icon: str = "",
) -> None:
    """Display a compact page header for presentation mode."""

    displayed_title = (
        f"{icon} {title}"
        if icon
        else title
    )

    st.markdown(
        f"## {displayed_title}"
    )

    st.caption(subtitle)