# Component: Kpi Cards
from collections.abc import Sequence

import streamlit as st


MetricEntry = tuple[str, str] | tuple[str, str, str]


def show_kpi_cards(
    metrics: Sequence[MetricEntry],
) -> None:
    """
    Display KPI cards.

    Entries may be:
    (label, value)
    or
    (label, value, help_text)
    """

    if not metrics:
        return

    columns = st.columns(len(metrics))

    for column, metric in zip(columns, metrics):
        label = metric[0]
        value = metric[1]
        help_text = metric[2] if len(metric) == 3 else None

        column.metric(
            label=label,
            value=value,
            help=help_text,
        )
