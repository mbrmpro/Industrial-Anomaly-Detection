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
    PATCHCORE_METRICS_PATH,
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
    title="Model Comparison",
    subtitle=(
        "Comparison of reconstruction-based CAE and "
        "feature-memory-based PatchCore anomaly detection."
    ),
    icon="⚖️",
)


# =========================================================
# PATH VALIDATION
# =========================================================

required_paths = {
    "PatchCore metrics": PATCHCORE_METRICS_PATH,
    "CAE metrics": CAE_METRICS_PATH,
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
def load_metrics(
    path: str | Path,
) -> pd.DataFrame:
    return pd.read_csv(path)


try:
    patchcore_df = load_metrics(
        PATCHCORE_METRICS_PATH
    )

    cae_df = load_metrics(
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

patchcore_mean_auroc = float(
    patchcore_df["image_auroc"].mean()
)

patchcore_mean_f1 = float(
    patchcore_df["image_f1"].mean()
)

cae_mean_auroc = float(
    cae_df["auroc"].mean()
)

cae_mean_f1 = float(
    cae_df["f1"].mean()
)


show_kpi_cards(
    [
        (
            "PatchCore AUROC",
            f"{patchcore_mean_auroc:.3f}",
        ),
        (
            "CAE AUROC",
            f"{cae_mean_auroc:.3f}",
        ),
        (
            "PatchCore F1",
            f"{patchcore_mean_f1:.3f}",
        ),
        (
            "CAE F1",
            f"{cae_mean_f1:.3f}",
        ),
    ]
)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Learning Principle",
        "Architecture",
        "Performance",
        "Final Selection",
    ]
)


# =========================================================
# TAB 1 — LEARNING PRINCIPLE
# =========================================================

with tab1:

    cae_col, patchcore_col = st.columns(2)

    with cae_col:

        st.markdown("#### Convolutional Autoencoder")

        st.markdown(
            """
            The CAE learns to reconstruct normal images.

            Anomalies are identified when the reconstruction error
            exceeds the threshold learned from normal training scores.
            """
        )

        st.code(
            """
Normal Image
    ↓
Encoder
    ↓
Latent Representation
    ↓
Decoder
    ↓
Reconstruction
    ↓
Reconstruction Error
            """,
            language="text",
        )

    with patchcore_col:

        st.markdown("#### PatchCore")

        st.markdown(
            """
            PatchCore stores representative local features extracted
            from normal images.

            Anomalies are identified through large nearest-neighbour
            distances from the normal feature memory bank.
            """
        )

        st.code(
            """
Normal Images
    ↓
Wide-ResNet-50-2
    ↓
Local Patch Features
    ↓
Coreset Memory Bank
    ↓
Nearest-Neighbour Distance
    ↓
Anomaly Score and Map
            """,
            language="text",
        )

    principle_df = pd.DataFrame(
        {
            "Aspect": [
                "Training data",
                "Anomaly definition",
                "Decoder",
                "Memory bank",
                "Localization",
            ],
            "CAE": [
                "Normal images only",
                "High reconstruction error",
                "Required",
                "Not used",
                "Reconstruction-error map",
            ],
            "PatchCore": [
                "Normal images only",
                "Large feature-space distance",
                "Not required",
                "Coreset memory bank",
                "Patch-level anomaly map",
            ],
        }
    )

    st.dataframe(
        principle_df,
        use_container_width=True,
        hide_index=True,
        height=250,
    )


# =========================================================
# TAB 2 — ARCHITECTURE
# =========================================================

with tab2:

    architecture_df = pd.DataFrame(
        {
            "Property": [
                "Model type",
                "Representation",
                "Training objective",
                "Backbone",
                "Category-specific model",
                "Primary anomaly signal",
                "Interpretability",
            ],
            "CAE": [
                "Encoder-decoder network",
                "Learned latent representation",
                "Reconstruct normal images",
                "Developed from scratch",
                "Yes",
                "Reconstruction score",
                "Reconstruction error and Grad-CAM",
            ],
            "PatchCore": [
                "Feature-memory method",
                "Pretrained local embeddings",
                "Construct normal memory bank",
                "Wide-ResNet-50-2",
                "Yes",
                "Nearest-neighbour distance",
                "Anomaly map and predicted mask",
            ],
        }
    )

    st.dataframe(
        architecture_df,
        use_container_width=True,
        hide_index=True,
        height=330,
    )

    left_info, right_info = st.columns(2)

    with left_info:
        st.info(
            "The CAE learns encoder and decoder parameters directly "
            "from normal training images."
        )

    with right_info:
        st.info(
            "PatchCore mainly extracts pretrained features and builds "
            "a compressed normal-feature memory bank."
        )


# =========================================================
# TAB 3 — PERFORMANCE
# =========================================================

with tab3:

    comparison_df = pd.DataFrame(
        {
            "Model": [
                "PatchCore",
                "CAE",
            ],
            "Mean AUROC": [
                patchcore_mean_auroc,
                cae_mean_auroc,
            ],
            "Mean F1": [
                patchcore_mean_f1,
                cae_mean_f1,
            ],
            "Reported Categories": [
                len(patchcore_df),
                len(cae_df),
            ],
        }
    )

    chart_col, table_col = st.columns(
        [1.15, 1]
    )

    with chart_col:

        figure, axis = plt.subplots(
            figsize=(
                FIGURE_WIDTH,
                FIGURE_HEIGHT,
            ),
            dpi=FIGURE_DPI,
        )

        x_positions = range(
            len(comparison_df)
        )

        bar_width = 0.34

        axis.bar(
            [
                position - bar_width / 2
                for position in x_positions
            ],
            comparison_df["Mean AUROC"],
            width=bar_width,
            label="Mean AUROC",
        )

        axis.bar(
            [
                position + bar_width / 2
                for position in x_positions
            ],
            comparison_df["Mean F1"],
            width=bar_width,
            label="Mean F1",
        )

        axis.set_xticks(
            list(x_positions),
            comparison_df["Model"],
        )

        axis.set_ylim(
            0,
            1.05,
        )

        axis.set_ylabel(
            "Score",
            fontsize=9,
            fontweight="bold",
        )

        axis.set_title(
            "Mean Image-Level Performance",
            fontsize=11,
            fontweight="bold",
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
            frameon=False,
            fontsize=8,
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

        st.dataframe(
            comparison_df.style.format(
                {
                    "Mean AUROC": "{:.3f}",
                    "Mean F1": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.warning(
            """
            PatchCore reports results for all 15 categories.

            The CAE summary contains 14 reported categories because
            no toothbrush result was supplied.
            """
        )

    st.markdown("#### Bottle Case Study")

    bottle_patchcore = patchcore_df.loc[
        patchcore_df["category"] == "bottle"
    ]

    bottle_cae = cae_df.loc[
        cae_df["category"] == "bottle"
    ]

    if (
        not bottle_patchcore.empty
        and not bottle_cae.empty
    ):

        bottle_df = pd.DataFrame(
            {
                "Model": [
                    "PatchCore",
                    "CAE",
                ],
                "AUROC": [
                    bottle_patchcore.iloc[0][
                        "image_auroc"
                    ],
                    bottle_cae.iloc[0]["auroc"],
                ],
                "F1": [
                    bottle_patchcore.iloc[0][
                        "image_f1"
                    ],
                    bottle_cae.iloc[0]["f1"],
                ],
            }
        )

        st.dataframe(
            bottle_df.style.format(
                {
                    "AUROC": "{:.4f}",
                    "F1": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# TAB 4 — FINAL MODEL SELECTION
# =========================================================

with tab4:

    strengths_df = pd.DataFrame(
        {
            "Aspect": [
                "Primary strength",
                "Localization",
                "Interpretability",
                "Main limitation",
                "Final project role",
            ],
            "CAE": [
                "Developed from scratch",
                "Indirect reconstruction-error map",
                "Reconstruction and Grad-CAM",
                "May reconstruct anomalies accurately",
                "Interpretable baseline",
            ],
            "PatchCore": [
                "Strong anomaly detection",
                "Dedicated anomaly map and mask",
                "Distance from normal memory",
                "Pretrained backbone and memory bank",
                "Primary anomaly detector",
            ],
        }
    )

    st.dataframe(
        strengths_df,
        use_container_width=True,
        hide_index=True,
        height=280,
    )

    st.success(
        """
        **Final Decision**

        PatchCore was selected as the primary anomaly detector because
        it achieved stronger and more consistent image-level performance
        and provides quantitatively evaluated pixel-level localization.

        The CAE remains an interpretable, fully self-developed baseline
        for reconstruction-based comparison.
        """
    )


# =========================================================
# TAKEAWAY
# =========================================================

show_takeaway(
    (
        "The CAE explains anomalies as reconstruction failures, while "
        "PatchCore identifies deviations in pretrained local feature space. "
        "The experimental results support PatchCore as the more reliable "
        "detector for the final inspection pipeline."
    )
)