import textwrap

import streamlit as st


def _render_html(html: str) -> None:
    """Render HTML without Markdown indentation problems."""

    st.markdown(
        textwrap.dedent(html).strip(),
        unsafe_allow_html=True,
    )


def _box(
    text: str,
    border_color: str = "#2563EB",
    background: str = "#FFFFFF",
    text_color: str = "#0F172A",
) -> None:
    """Render one compact workflow box."""

    _render_html(
        f"""
        <div style="
            width: 78%;
            margin: 0 auto;
            padding: 0.45rem 0.60rem;
            border: 1.5px solid {border_color};
            border-radius: 8px;
            background: {background};
            color: {text_color};
            text-align: center;
            font-size: 0.88rem;
            font-weight: 650;
            line-height: 1.2;
        ">
            {text}
        </div>
        """
    )


def _arrow(
    color: str = "#2563EB",
) -> None:
    """Render one compact vertical arrow."""

    _render_html(
        f"""
        <div style="
            height: 24px;
            margin: 0;
            text-align: center;
            color: {color};
            font-size: 1.35rem;
            line-height: 24px;
        ">
            ↓
        </div>
        """
    )


def show_pipeline_diagram() -> None:
    """Display the complete inspection pipeline."""

    _box("Input Image")
    _arrow()

    _box("DINOv2 ViT-L/14")
    _arrow()

    _box("2,048-D Hybrid Feature")
    _arrow()

    _box("Logistic Regression Object Router")
    _arrow()

    _box("Object Category")
    _arrow()

    _box("Category-Specific PatchCore")
    _arrow()

    _box(
        "Normal or Anomalous?",
        border_color="#8B5CF6",
        background="#F5F3FF",
    )

        # =====================================================
    # NORMAL / ANOMALOUS BRANCH
    # Native Streamlit components avoid HTML rendering bugs.
    # =====================================================

    branch_left, branch_right = st.columns(
        2,
        gap="small",
    )

    with branch_left:

        st.markdown(
            "<div style='text-align:center; color:#16A34A; "
            "font-size:1.25rem;'>↙</div>",
            unsafe_allow_html=True,
        )

        with st.container(border=True):

            st.markdown(
                """
                <div style="
                    text-align:center;
                    color:#166534;
                    font-weight:700;
                    line-height:1.5;
                ">
                    Normal<br>
                    Final Result
                </div>
                """,
                unsafe_allow_html=True,
            )

    with branch_right:

        st.markdown(
            "<div style='text-align:center; color:#DC2626; "
            "font-size:1.25rem;'>↘</div>",
            unsafe_allow_html=True,
        )

        with st.container(border=True):

            st.markdown(
                """
                <div style="
                    text-align:center;
                    color:#991B1B;
                    font-weight:700;
                    line-height:1.5;
                ">
                    Anomalous<br>
                    Heatmap
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div style='text-align:center; color:#EA580C; "
            "font-size:1.25rem;'>↓</div>",
            unsafe_allow_html=True,
        )

        with st.container(border=True):

            st.markdown(
                """
                <div style="
                    text-align:center;
                    color:#92400E;
                    font-weight:700;
                ">
                    Polynomial SVM
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div style='text-align:center; color:#EA580C; "
            "font-size:1.25rem;'>↓</div>",
            unsafe_allow_html=True,
        )

        with st.container(border=True):

            st.markdown(
                """
                <div style="
                    text-align:center;
                    color:#92400E;
                    font-weight:700;
                ">
                    Defect Type
                </div>
                """,
                unsafe_allow_html=True,
            )