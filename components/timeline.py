from collections.abc import Sequence

import streamlit as st


TimelineStep = tuple[str, str, str, str]


def show_timeline(
    steps: Sequence[TimelineStep],
    columns_per_row: int = 5,
) -> None:
    """
    Display a compact presentation workflow.

    The page path remains part of the data structure for future use,
    but navigation is handled through the Streamlit sidebar.
    """

    if not steps:
        return

    columns_per_row = max(
        1,
        min(columns_per_row, 5),
    )

    step_number = 1

    for start_index in range(
        0,
        len(steps),
        columns_per_row,
    ):

        row_steps = steps[
            start_index:
            start_index + columns_per_row
        ]

        columns = st.columns(
            columns_per_row
        )

        for column, step in zip(
            columns,
            row_steps,
        ):

            icon, title, description, _page_path = step

            with column:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        f"#### {step_number}. {icon} {title}"
                    )

                    st.caption(
                        description
                    )

            step_number += 1