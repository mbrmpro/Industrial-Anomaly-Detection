import streamlit as st

from components.footer import show_footer
from components.hero import show_hero
from components.info_box import show_info_box
from components.kpi_cards import show_kpi_cards
from components.sidebar import show_sidebar
from components.takeaway import show_takeaway
from components.timeline import show_timeline

from utils.config import PROJECT_NAME
from utils.style_loader import load_css

from components.pipeline_diagram import show_pipeline_diagram


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title=PROJECT_NAME,
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# GLOBAL APPLICATION ELEMENTS
# =========================================================

load_css()
show_sidebar()


# =========================================================
# HERO
# =========================================================

show_hero(
    title="Industrial Visual Quality Inspection",
    subtitle=(
        "One-Class Anomaly Detection and Defect Classification "
        "for Industrial Components"
    ),
    technologies=(
        "MVTec AD • PatchCore • DINOv2 • "
        "Convolutional Autoencoder • OpenVINO"
    ),
)


# =========================================================
# PROJECT KPIs
# =========================================================

show_kpi_cards(
    [
        (
            "Dataset Categories",
            "15",
            "Industrial object and texture categories in MVTec AD.",
        ),
        (
            "Primary Detector",
            "PatchCore",
            "Selected as the final anomaly-detection method.",
        ),
        (
            "Mean Image AUROC",
            "0.983",
            "PatchCore mean image-level AUROC across 15 categories.",
        ),
        (
            "Deployment",
            "OpenVINO",
            "Category-specific models used by the live inspection page.",
        ),
    ]
)


# =========================================================
# PROJECT OVERVIEW TABS
# =========================================================

problem_tab, pipeline_tab, models_tab = st.tabs(
    [
        "Industrial Problem",
        "Final Pipeline",
        "Model Selection",
    ]
)


# =========================================================
# TAB 1 — INDUSTRIAL PROBLEM
# =========================================================

with problem_tab:

    problem_col, objective_col = st.columns(2)

    with problem_col:

        st.markdown("#### The Challenge")

        st.markdown(
            """
            Industrial defects such as cracks, scratches, contamination
            and structural damage are rare and heterogeneous.

            Large labelled defect datasets are usually unavailable.
            The inspection system must therefore learn normal appearance
            and identify previously unseen deviations.
            """
        )

    with objective_col:

        st.markdown("#### Project Objective")

        st.markdown(
            """
            The complete system performs three connected tasks:

            1. identify the industrial object category;
            2. detect and localize an anomaly;
            3. assign a defect type when an anomaly is present.
            """
        )

    show_info_box(
        title="One-Class Learning",
        text=(
            "Normal images are used to train the anomaly detectors. "
            "Defective test images and ground-truth masks are reserved "
            "for detection and localization evaluation."
        ),
        box_type="info",
    )


# =========================================================
# TAB 2 — FINAL PIPELINE
# =========================================================

with pipeline_tab:

    workflow_col, explanation_col = st.columns(
        [1.05, 1]
    )

    with workflow_col:

        st.markdown("#### End-to-End Inspection Workflow")

        show_pipeline_diagram()

    with explanation_col:

        st.markdown("#### Deployed Components")

        pipeline_df = {
            "Object routing": (
                "DINOv2 hybrid features and Logistic Regression"
            ),
            "Anomaly detection": (
                "Category-specific PatchCore model"
            ),
            "Localization": (
                "PatchCore anomaly map and overlay"
            ),
            "Defect classification": (
                "Category-specific Polynomial SVM"
            ),
            "Inference engine": (
                "OpenVINO"
            ),
        }

        for component, method in pipeline_df.items():

            st.markdown(
                f"**{component}:** {method}"
            )

        st.info(
            "The defect classifier runs only when PatchCore "
            "detects an anomaly."
        )


# =========================================================
# TAB 3 — MODEL SELECTION
# =========================================================

with models_tab:

    cae_col, patchcore_col = st.columns(2)

    with cae_col:

        st.markdown("#### Convolutional Autoencoder")

        st.markdown(
            """
            **Role:** interpretable baseline

            - developed from scratch;
            - learns to reconstruct normal images;
            - uses reconstruction error for anomaly detection;
            - supports reconstruction maps and Grad-CAM.
            """
        )

    with patchcore_col:

        st.markdown("#### PatchCore")

        st.markdown(
            """
            **Role:** primary anomaly detector

            - uses pretrained local patch features;
            - builds a normal-feature memory bank;
            - produces an anomaly score and anomaly map;
            - supports pixel-level localization.
            """
        )

    show_info_box(
        title="Final Decision",
        text=(
            "PatchCore was selected because it achieved stronger and "
            "more consistent image-level performance and provided "
            "quantitatively evaluated anomaly localization."
        ),
        box_type="success",
    )


# =========================================================
# PRESENTATION WORKFLOW
# =========================================================

st.markdown("## 🧭 Presentation Workflow")

st.caption(
    "Open each chapter in report order or start directly with the live demo."
)

show_timeline(
    [
        (
            "📦",
            "Dataset",
            (
                "MVTec AD categories, train/test structure, "
                "defect types and masks."
            ),
            "pages/02_Dataset.py",
        ),
        (
            "📊",
            "EDA",
            (
                "Image counts, defect diversity, severity, "
                "area and RGB analysis."
            ),
            "pages/03_EDA.py",
        ),
        (
            "⚙️",
            "Preprocessing",
            (
                "Resize, normalization and object-safe "
                "offline augmentation."
            ),
            "pages/04_Preprocessing.py",
        ),
        (
            "🧠",
            "CAE Baseline",
            (
                "Reconstruction-based one-class "
                "anomaly-detection baseline."
            ),
            "pages/05_CAE_Baseline.py",
        ),
        (
            "🔬",
            "Feature Engineering",
            (
                "DINOv2 hybrid features for routing "
                "and defect classification."
            ),
            "pages/06_Feature_Engineering.py",
        ),
        (
            "🚀",
            "PatchCore",
            (
                "Memory-bank anomaly detection, "
                "localization and OpenVINO deployment."
            ),
            "pages/07_PatchCore.py",
        ),
        (
            "⚖️",
            "Model Comparison",
            (
                "Comparison of reconstruction-based CAE "
                "and feature-memory PatchCore."
            ),
            "pages/08_Model_Comparison.py",
        ),
        (
            "📈",
            "Evaluation",
            (
                "Image-, pixel- and defect-classification "
                "performance."
            ),
            "pages/09_Evaluation.py",
        ),
        (
            "✅",
            "Conclusions",
            (
                "Findings, limitations and future work."
            ),
            "pages/10_Conclusions.py",
        ),
        (
            "🔍",
            "Live Inspection",
            (
                "Run the complete deployed inspection pipeline."
            ),
            "pages/11_Live_Inspection.py",
        ),
    ],
    columns_per_row=5,
)


# =========================================================
# FINAL TAKEAWAY
# =========================================================

show_takeaway(
    (
        "The project delivers an end-to-end industrial inspection "
        "pipeline combining DINOv2 object routing, category-specific "
        "PatchCore anomaly detection, spatial localization and "
        "conditional defect-type classification."
    )
)


# =========================================================
# FOOTER
# =========================================================

show_footer()