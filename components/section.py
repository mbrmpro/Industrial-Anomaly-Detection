# Component: Section
import streamlit as st


def show_section(
    title: str,
    description: str | None = None,
    icon: str = "",
    divider: bool = False,
) -> None:
    """Display a standardized section heading."""

    if divider:
        st.divider()

    displayed_title = f"{icon} {title}" if icon else title

    st.subheader(displayed_title)

    if description:
        st.markdown(
            f"""
            <p style="
                color: #64748B;
                margin-top: -0.35rem;
                margin-bottom: 1rem;
            ">
                {description}
            </p>
            """,
            unsafe_allow_html=True,
        )
