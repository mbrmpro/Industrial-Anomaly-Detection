from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from components.kpi_cards import show_kpi_cards
from components.page_header import show_page_header
from components.takeaway import show_takeaway

from utils.config import (
    CAE_METRICS_PATH,
    FIGURE_DPI,
    FIGURE_HEIGHT,
    FIGURE_WIDTH,
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
    title="Convolutional Autoencoder Baseline",
    subtitle=(
        "Reconstruction-based one-class anomaly detection "
        "developed from scratch as an interpretable baseline."
    ),
    icon="🧠",
)


# =========================================================
# LOAD RESULTS
# =========================================================

if not Path(CAE_METRICS_PATH).exists():
    st.error(
        "CAE metrics file not found:\n\n"
        f"`{CAE_METRICS_PATH}`"
    )
    st.stop()


@st.cache_data
def load_cae_metrics(
    path: str | Path,
) -> pd.DataFrame:
    """Load the precomputed category-wise CAE metrics."""

    dataframe = pd.read_csv(path)

    required_columns = {
        "category",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auroc",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "CAE metrics file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return dataframe


try:
    cae_df = load_cae_metrics(
        CAE_METRICS_PATH
    )

except (
    FileNotFoundError,
    ValueError,
    OSError,
) as error:
    st.exception(error)
    st.stop()


# =========================================================
# SUMMARY METRICS
# =========================================================

mean_accuracy = float(
    cae_df["accuracy"].mean()
)

mean_f1 = float(
    cae_df["f1"].mean()
)

mean_auroc = float(
    cae_df["auroc"].mean()
)


show_kpi_cards(
    [
        (
            "Reported Categories",
            str(len(cae_df)),
        ),
        (
            "Mean Accuracy",
            f"{mean_accuracy:.3f}",
        ),
        (
            "Mean F1",
            f"{mean_f1:.3f}",
        ),
        (
            "Mean AUROC",
            f"{mean_auroc:.3f}",
        ),
    ]
)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Principle",
        "Architecture",
        "Training & Threshold",
        "Results",
        "Interpretability",
    ]
)


# =========================================================
# TAB 1 — PRINCIPLE
# =========================================================

with tab1:

    left_col, right_col = st.columns(
        [1.15, 1]
    )

    with left_col:

        st.markdown("#### One-Class Reconstruction")

        st.markdown(
            """
            The CAE is trained using normal images only.

            It learns to compress and reconstruct the normal visual
            appearance of each category. Defective structures are
            expected to produce larger reconstruction errors.
            """
        )

        st.info(
            "The model does not learn individual defect classes. "
            "It learns normal appearance and detects deviations."
        )

    with right_col:

        st.markdown("#### Model Workflow")

        st.code(
            """
Input Image
    ↓
Encoder
    ↓
Latent Representation
    ↓
Decoder
    ↓
Reconstructed Image
    ↓
Reconstruction Error
    ↓
Normal / Anomalous
            """,
            language="text",
        )


# =========================================================
# TAB 2 — ARCHITECTURE
# =========================================================

with tab2:

    architecture_df = pd.DataFrame(
        {
            "Stage": [
                "Input",
                "Encoder 1",
                "Encoder 2",
                "Encoder 3",
                "Encoder 4",
                "Latent Space",
                "Decoder",
                "Output",
            ],
            "Configuration": [
                "256 × 256 RGB",
                "32 filters",
                "64 filters",
                "128 filters",
                "256 filters",
                "Compressed representation",
                "Transposed convolutions",
                "3-channel sigmoid output",
            ],
            "Purpose": [
                "Standardized input",
                "Edges and colour transitions",
                "Local textures",
                "Intermediate structures",
                "High-level structures",
                "Normal feature representation",
                "Spatial reconstruction",
                "Pixel values in [0, 1]",
            ],
        }
    )

    st.dataframe(
        architecture_df,
        use_container_width=True,
        hide_index=True,
        height=320,
    )

    insight_col, limitation_col = st.columns(2)

    with insight_col:

        st.info(
            """
            **Skip Connections**

            U-Net-like skip connections preserve spatial information
            and can improve reconstruction quality.
            """
        )

    with limitation_col:

        st.warning(
            """
            **Risk**

            A decoder that is too powerful may reconstruct anomalies
            accurately and reduce the separation between normal and
            defective scores.
            """
        )


# =========================================================
# TAB 3 — TRAINING AND THRESHOLD
# =========================================================

with tab3:

    config_col, threshold_col = st.columns(
        [1.15, 1]
    )

    with config_col:

        training_df = pd.DataFrame(
            {
                "Parameter": [
                    "Training data",
                    "Input resolution",
                    "Batch size",
                    "Maximum epochs",
                    "Optimizer",
                    "Learning rate",
                    "Weight decay",
                    "Validation split",
                    "Early stopping",
                    "LR reduction",
                ],
                "Value": [
                    "Normal train/good only",
                    "256 × 256",
                    "16",
                    "50",
                    "AdamW",
                    "1e-4",
                    "1e-5",
                    "10%",
                    "8 epochs",
                    "Factor 0.5 after 3 epochs",
                ],
            }
        )

        st.dataframe(
            training_df,
            use_container_width=True,
            hide_index=True,
            height=365,
        )

    with threshold_col:

        st.markdown("#### Decision Threshold")

        st.latex(
            r"""
            \tau =
            Q_{0.95}
            \left(
            \{S(x_i)\mid x_i\in D_{\mathrm{train}}\}
            \right)
            """
        )

        st.markdown(
            """
            The threshold is estimated from reconstruction scores
            of normal training images.

            - \(S(x) \leq \tau\): **Normal**
            - \(S(x) > \tau\): **Anomalous**
            """
        )

        st.success(
            "The threshold is determined without using defective "
            "test labels, preserving the one-class setting."
        )


# =========================================================
# TAB 4 — RESULTS
# =========================================================

with tab4:

    selected_category = st.selectbox(
        "Select category",
        options=sorted(
            cae_df["category"].tolist()
        ),
        key="cae_result_category",
    )

    selected_row = cae_df.loc[
        cae_df["category"]
        == selected_category
    ].iloc[0]

    result1, result2, result3, result4, result5 = (
        st.columns(5)
    )

    result1.metric(
        "Accuracy",
        f"{selected_row['accuracy']:.3f}",
    )

    result2.metric(
        "Precision",
        f"{selected_row['precision']:.3f}",
    )

    result3.metric(
        "Recall",
        f"{selected_row['recall']:.3f}",
    )

    result4.metric(
        "F1",
        f"{selected_row['f1']:.3f}",
    )

    result5.metric(
        "AUROC",
        f"{selected_row['auroc']:.3f}",
    )

    chart_col, table_col = st.columns(
        [1.25, 1]
    )

    with chart_col:

        chart_df = cae_df.sort_values(
            "auroc",
            ascending=True,
        )

        figure, axis = plt.subplots(
            figsize=(
                FIGURE_WIDTH,
                FIGURE_HEIGHT,
            ),
            dpi=FIGURE_DPI,
        )

        axis.barh(
            chart_df["category"],
            chart_df["auroc"],
        )

        axis.set_xlim(
            0,
            1.02,
        )

        axis.set_xlabel(
            "AUROC",
            fontsize=9,
            fontweight="bold",
        )

        axis.set_ylabel(
            "Category",
            fontsize=9,
            fontweight="bold",
        )

        axis.set_title(
            "CAE Image-Level AUROC by Category",
            fontsize=11,
            fontweight="bold",
        )

        axis.tick_params(
            axis="both",
            labelsize=8,
        )

        axis.grid(
            axis="x",
            linestyle="--",
            alpha=0.3,
        )

        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        figure.tight_layout(
            pad=0.5
        )

        st.pyplot(
            figure,
            use_container_width=True,
        )

        plt.close(figure)

    with table_col:

        visible_columns = [
            column
            for column in [
                "category",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "auroc",
            ]
            if column in cae_df.columns
        ]

        display_df = (
            cae_df[visible_columns]
            .rename(
                columns={
                    "category": "Category",
                    "accuracy": "Accuracy",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1": "F1",
                    "auroc": "AUROC",
                }
            )
        )

        st.dataframe(
            display_df.style.format(
                {
                    "Accuracy": "{:.3f}",
                    "Precision": "{:.3f}",
                    "Recall": "{:.3f}",
                    "F1": "{:.3f}",
                    "AUROC": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
            height=360,
        )

    st.info(
        """
        Category performance varies because reconstruction error is
        influenced by object alignment, normal texture variability,
        defect size and decoder capacity.
        """
    )


# =========================================================
# TAB 5 — INTERPRETABILITY
# =========================================================

with tab5:

    explanation_col, layers_col = st.columns(
        [1, 1.15]
    )

    with explanation_col:

        st.markdown("#### Reconstruction Error")

        st.markdown(
            """
            The absolute difference between the original and reconstructed
            image highlights regions that the CAE cannot reproduce well.
            """
        )

        st.markdown("#### Grad-CAM")

        st.markdown(
            """
            Grad-CAM activation maps indicate which spatial regions
            influence internal convolutional representations.
            """
        )

        st.info(
            "Reconstruction-error maps and Grad-CAM are complementary: "
            "one shows reconstruction failure, the other model activation."
        )

    with layers_col:

        interpretability_df = pd.DataFrame(
            {
                "Network Level": [
                    "32-filter layer",
                    "64-filter layer",
                    "128-filter layer",
                    "256/512-filter layers",
                    "Final RGB layer",
                ],
                "Observed Behaviour": [
                    "Edges and intensity transitions",
                    "Local component structures",
                    "Localized defect responses",
                    "Broader semantic activations",
                    "Object contour and reconstruction",
                ],
            }
        )

        st.dataframe(
            interpretability_df,
            use_container_width=True,
            hide_index=True,
            height=280,
        )


# =========================================================
# TAKEAWAY
# =========================================================

show_takeaway(
    (
        "The CAE provides a transparent, self-developed one-class "
        "baseline. It performs well for clearly visible structural "
        "defects, but its robustness decreases when normal variability "
        "and subtle defects produce overlapping reconstruction scores."
    )
)