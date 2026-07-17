# Component: Info Box
from typing import Literal

import streamlit as st


BoxType = Literal[
    "info",
    "success",
    "warning",
    "error",
]


def show_info_box(
    title: str,
    text: str,
    box_type: BoxType = "info",
) -> None:
    """Display a standardized message box."""

    message = f"**{title}**\n\n{text}"

    functions = {
        "info": st.info,
        "success": st.success,
        "warning": st.warning,
        "error": st.error,
    }

    functions[box_type](message)
