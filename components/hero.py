import streamlit as st


def show_hero(
    title: str,
    subtitle: str,
    technologies: str | None = None,
) -> None:
    """Display a stable Streamlit hero section without custom HTML."""

    with st.container(border=True):

        st.title(title)

        st.markdown(
            f"### {subtitle}"
        )

        if technologies:
            st.info(technologies)