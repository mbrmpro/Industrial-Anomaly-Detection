from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from components.page_header import show_page_header
from components.kpi_cards import show_kpi_cards
from components.takeaway import show_takeaway

from utils.config import (
    PATCHCORE_METRICS_PATH,
    CAE_METRICS_PATH,
    DEFECT_CLASSIFIER_REPORTS_PATH,
    FIGURE_DPI,
)

from utils.model_metrics import (
    load_defect_classifier_reports,
)

from components.sidebar import show_sidebar
from utils.style_loader import load_css

# =========================================================
# GLOBAL APPLICATION ELEMENTS
# =========================================================
load_css()
show_sidebar()
# =========================================================
# PAGE HEADER
# =========================================================

show_page_header(
    title="System Evaluation",
    subtitle=(
        "Quantitative evaluation of the complete industrial "
        "inspection pipeline."
    ),
    icon="📊",
)


# =========================================================
# PATH VALIDATION
# =========================================================

required_paths = {
    "PatchCore metrics": PATCHCORE_METRICS_PATH,
    "CAE metrics": CAE_METRICS_PATH,
    "Defect-classifier reports":
        DEFECT_CLASSIFIER_REPORTS_PATH,
}

for label, path in required_paths.items():
    if not Path(path).exists():
        st.error(
            f"{label} not found:\n\n`{path}`"
        )
        st.stop()


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_csv(
    path: str | Path,
) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_defect_reports(
    path: str | Path,
) -> pd.DataFrame:
    return load_defect_classifier_reports(path)


def show_compact_figure(
    figure,
    *,
    column_ratios=(1.2, 3.0, 1.2),
) -> None:
    """Render a Matplotlib figure in a centered, narrower Streamlit column."""

    _, chart_column, _ = st.columns(
        list(column_ratios)
    )

    with chart_column:
        st.pyplot(
            figure,
            use_container_width=True,
        )

    plt.close(figure)


try:
    patchcore_df = load_csv(
        PATCHCORE_METRICS_PATH
    )

    cae_df = load_csv(
        CAE_METRICS_PATH
    )

    defect_df = load_defect_reports(
        DEFECT_CLASSIFIER_REPORTS_PATH
    )

except Exception as error:
    st.exception(error)
    st.stop()


# =========================================================
# OVERALL METRICS
# =========================================================

router_accuracy = 1.00

patchcore_mean_auroc = (
    patchcore_df["image_auroc"].mean()
)

patchcore_mean_f1 = (
    patchcore_df["image_f1"].mean()
)

patchcore_mean_pixel_f1 = (
    patchcore_df["pixel_f1"].mean()
)

cae_mean_auroc = (
    cae_df["auroc"].mean()
)

cae_mean_f1 = (
    cae_df["f1"].mean()
)

defect_mean_macro_f1 = (
    defect_df["macro_f1_mean"]
    .dropna()
    .mean()
)


show_kpi_cards(
    [
        (
            "Router Accuracy",
            f"{router_accuracy:.3f}",
        ),
        (
            "PatchCore AUROC",
            f"{patchcore_mean_auroc:.3f}",
        ),
        (
            "CAE AUROC",
            f"{cae_mean_auroc:.3f}",
        ),
        (
            "Macro F1",
            f"{defect_mean_macro_f1:.3f}",
        ),
    ]
)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Overall Performance",
        "PatchCore Results",
        "CAE Results",
        "Category Comparison",
    ]
)


# =========================================================
# TAB 1 — OVERALL PERFORMANCE
# =========================================================

with tab1:



    summary_df = pd.DataFrame(
        {
            "Component": [
                "DINOv2 Object Router",
                "PatchCore",
                "Convolutional Autoencoder",
                "Defect-Type Classifier",
            ],
            "Primary Metric": [
                "Accuracy",
                "Mean Image AUROC",
                "Mean Image AUROC",
                "Mean Macro F1",
            ],
            "Result": [
                router_accuracy,
                patchcore_mean_auroc,
                cae_mean_auroc,
                defect_mean_macro_f1,
            ],
        }
    )

    st.dataframe(
        summary_df.style.format(
            {"Result": "{:.3f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    chart_df = summary_df.sort_values(
        "Result",
        ascending=True,
    )

    figure, axis = plt.subplots(
        figsize=(6.4, 3.4),
        dpi=FIGURE_DPI,
    )

    axis.barh(
        chart_df["Component"],
        chart_df["Result"],
    )

    axis.set_xlim(0, 1.05)

    axis.set_xlabel(
        "Score",
        fontweight="bold",
    )

    axis.set_title(
        "Overall Model Performance",
        fontsize=13,
        fontweight="bold",
    )

    axis.tick_params(
        axis="both",
        labelsize=9,
    )

    axis.grid(
        axis="x",
        linestyle="--",
        alpha=0.3,
    )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    figure.tight_layout()

    show_compact_figure(
        figure,
    )

    st.info(
        """
        The DINOv2 router correctly separated the 15 MVTec AD
        categories on the available evaluation data.

        PatchCore achieved the strongest overall anomaly-detection
        performance. The convolutional autoencoder provided a useful
        interpretable baseline but showed substantially more variation
        across categories.

        The defect classifiers achieved strong average performance,
        although their stability is limited by the small number of
        independent defect images.
        """
    )


# =========================================================
# TAB 2 — PATCHCORE RESULTS
# =========================================================

with tab2:



    pc1, pc2, pc3, pc4, pc5 = (
        st.columns(5)
    )

    pc1.metric(
        "Mean Image AUROC",
        f"{patchcore_mean_auroc:.3f}",
    )

    pc2.metric(
        "Mean Image F1",
        f"{patchcore_mean_f1:.3f}",
    )

    pc3.metric(
        "Mean Pixel AUROC",
        f"{patchcore_df['pixel_auroc'].mean():.3f}",
    )

    pc4.metric(
        "Mean Pixel F1",
        f"{patchcore_mean_pixel_f1:.3f}",
    )

    pc5.metric(
        "Mean AUPRO",
        f"{patchcore_df['aupro'].mean():.3f}",
    )

    patchcore_display = patchcore_df.rename(
        columns={
            "category": "Category",
            "aupro": "AUPRO",
            "image_auroc": "Image AUROC",
            "pixel_auroc": "Pixel AUROC",
            "image_f1": "Image F1",
            "pixel_f1": "Pixel F1",
        }
    )

    st.dataframe(
        patchcore_display.style.format(
            {
                "AUPRO": "{:.2f}",
                "Image AUROC": "{:.2f}",
                "Pixel AUROC": "{:.2f}",
                "Image F1": "{:.2f}",
                "Pixel F1": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    chart_df = patchcore_df.sort_values(
        "image_auroc",
        ascending=True,
    )

    figure, axis = plt.subplots(
        figsize=(7.2, 5.0),
        dpi=FIGURE_DPI,
    )

    axis.barh(
        chart_df["category"],
        chart_df["image_auroc"],
    )

    axis.set_xlim(0.85, 1.01)

    axis.set_xlabel(
        "Image AUROC",
        fontweight="bold",
    )

    axis.set_ylabel(
        "Category",
        fontweight="bold",
    )

    axis.set_title(
        "PatchCore Image-Level AUROC by Category",
        fontsize=13,
        fontweight="bold",
    )

    axis.tick_params(
        axis="both",
        labelsize=9,
    )

    axis.grid(
        axis="x",
        linestyle="--",
        alpha=0.3,
    )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    figure.tight_layout()

    show_compact_figure(
        figure,
    )

    st.warning(
        """
        Image-level detection is consistently strong. Exact localization
        is more difficult, as shown by the lower mean pixel F1-score.
        Thin or irregular defects can be detected correctly but still
        produce an imperfect thresholded segmentation boundary.
        """
    )


# =========================================================
# TAB 3 — CAE RESULTS
# =========================================================

with tab3:



    cae1, cae2, cae3 = st.columns(3)

    cae1.metric(
        "Mean Accuracy",
        f"{cae_df['accuracy'].mean():.3f}",
    )

    cae2.metric(
        "Mean F1",
        f"{cae_mean_f1:.3f}",
    )

    cae3.metric(
        "Mean AUROC",
        f"{cae_mean_auroc:.3f}",
    )

    cae_display = cae_df.rename(
        columns={
            "category": "Category",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "auroc": "AUROC",
        }
    )

    visible_columns = [
        column
        for column in [
            "Category",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "AUROC",
        ]
        if column in cae_display.columns
    ]

    cae_display = cae_display[
        visible_columns
    ]

    st.dataframe(
        cae_display.style.format(
            {
                column: "{:.3f}"
                for column in [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "AUROC",
                ]
                if column
                in cae_display.columns
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    chart_df = cae_df.sort_values(
        "auroc",
        ascending=True,
    )

    figure, axis = plt.subplots(
        figsize=(7.2, 5.0),
        dpi=FIGURE_DPI,
    )

    axis.barh(
        chart_df["category"],
        chart_df["auroc"],
    )

    axis.set_xlim(0, 1.02)

    axis.set_xlabel(
        "AUROC",
        fontweight="bold",
    )

    axis.set_ylabel(
        "Category",
        fontweight="bold",
    )

    axis.set_title(
        "CAE Image-Level AUROC by Category",
        fontsize=13,
        fontweight="bold",
    )

    axis.tick_params(
        axis="both",
        labelsize=9,
    )

    axis.grid(
        axis="x",
        linestyle="--",
        alpha=0.3,
    )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    figure.tight_layout()

    show_compact_figure(
        figure,
    )

    st.info(
        """
        The strongest CAE performance is observed for bottle and the
        second screw experiment. Carpet, grid and transistor remain
        more difficult because reconstruction error is influenced by
        normal texture variation, alignment and small defect size.
        """
    )


# =========================================================
# TAB 4 — CATEGORY COMPARISON
# =========================================================

with tab4:

    common_categories = sorted(
        set(
            patchcore_df["category"]
        ).intersection(
            set(cae_df["category"])
        )
    )

    selected_category = st.selectbox(
        "Select a category",
        options=common_categories,
        key="evaluation_category",
    )

    patchcore_row = patchcore_df.loc[
        patchcore_df["category"]
        == selected_category
    ].iloc[0]

    cae_row = cae_df.loc[
        cae_df["category"]
        == selected_category
    ].iloc[0]

    comparison_df = pd.DataFrame(
        {
            "Model": [
                "PatchCore",
                "CAE",
            ],
            "AUROC": [
                patchcore_row[
                    "image_auroc"
                ],
                cae_row["auroc"],
            ],
            "F1": [
                patchcore_row[
                    "image_f1"
                ],
                cae_row["f1"],
            ],
        }
    )

    figure, axis = plt.subplots(
        figsize=(5.2, 3.25),
        dpi=FIGURE_DPI,
    )

    x_positions = range(
        len(comparison_df)
    )

    width = 0.32

    axis.bar(
        [
            position - width / 2
            for position in x_positions
        ],
        comparison_df["AUROC"],
        width=width,
        label="AUROC",
    )

    axis.bar(
        [
            position + width / 2
            for position in x_positions
        ],
        comparison_df["F1"],
        width=width,
        label="F1",
    )

    axis.set_xticks(
        list(x_positions),
        comparison_df["Model"],
    )

    axis.set_ylim(0, 1.05)

    axis.set_ylabel(
        "Score",
        fontsize=9,
        fontweight="bold",
    )

    axis.set_title(
        f"Model Comparison — {selected_category}",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )

    axis.tick_params(
        axis="both",
        labelsize=8,
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
        fontsize=8,
    )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    figure.tight_layout()

    patchcore_column, chart_column, cae_column = st.columns(
        [1.05, 2.2, 1.05],
        gap="large",
    )

    with patchcore_column:
        st.markdown("### PatchCore")

        st.metric(
            "Image AUROC",
            f"{patchcore_row['image_auroc']:.3f}",
        )

        st.metric(
            "Image F1",
            f"{patchcore_row['image_f1']:.3f}",
        )

        st.metric(
            "Pixel F1",
            f"{patchcore_row['pixel_f1']:.3f}",
        )

    with chart_column:
        st.pyplot(
            figure,
            use_container_width=True,
        )

    with cae_column:
        st.markdown(
            "### Convolutional Autoencoder"
        )

        st.metric(
            "AUROC",
            f"{cae_row['auroc']:.3f}",
        )

        st.metric(
            "F1",
            f"{cae_row['f1']:.3f}",
        )

        st.metric(
            "Accuracy",
            f"{cae_row['accuracy']:.3f}",
        )

    plt.close(figure)

    st.caption(
        "PatchCore is the primary detector because it combines strong "
        "image-level performance with pixel-level localization. The CAE "
        "remains an interpretable reconstruction-based baseline."
    )
# =========================================================
# FINAL TAKEAWAY
# =========================================================

    show_takeaway(
    (
        "PatchCore provides the strongest overall anomaly-detection "
        "performance and is therefore the preferred detector. The "
        "convolutional autoencoder remains a useful interpretable "
        "baseline, while the DINOv2 router and category-specific SVMs "
        "complete the end-to-end inspection pipeline."
    )
    )

