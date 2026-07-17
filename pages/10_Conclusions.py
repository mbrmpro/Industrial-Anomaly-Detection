import streamlit as st

from components.page_header import show_page_header
from components.kpi_cards import show_kpi_cards
from components.takeaway import show_takeaway

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
    title="Project Conclusions",
    subtitle=(
        "Summary of the developed industrial anomaly-detection "
        "pipeline and the final project outcomes."
    ),
    icon="🏁",
)


# =========================================================
# KPI Cards
# =========================================================

show_kpi_cards(
    [
        (
            "PatchCore AUROC",
            "0.983",
        ),
        (
            "PatchCore F1",
            "0.972",
        ),
        (
            "Pixel F1",
            "0.581",
        ),
        (
            "Macro F1",
            "0.859",
        ),
    ]
)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Results",
        "Strengths",
        "Limitations",
        "Future Work",
    ]
)


# =========================================================
# TAB 1 — MAIN RESULTS
# =========================================================

with tab1:

    st.markdown(
        """
        The implemented inspection pipeline combines:

        1. a DINOv2-based object router;
        2. category-specific PatchCore anomaly detectors;
        3. category-specific SVM defect classifiers.

        The object router selects the correct product category.
        PatchCore determines whether the component is normal or anomalous
        and produces an anomaly map. If an anomaly is detected, the SVM
        assigns the most likely defect type.
        """
    )

    st.success(
        """
        PatchCore was selected as the primary anomaly detector because
        it provided the strongest and most consistent image-level
        performance together with quantitatively evaluated localization.
        """
    )


# =========================================================
# TAB 2 — STRENGTHS
# =========================================================

with tab2:


    st.markdown(
        """
        - The complete system follows a realistic one-class learning setup.
        - PatchCore requires only normal training images.
        - The anomaly map provides spatial localization of defects.
        - The DINOv2 router selects the category-specific detector.
        - The defect classifier adds a diagnostic defect label.
        - The CAE provides an interpretable reconstruction-based baseline.
        - The final models can be used in a live Streamlit inspection workflow.
        """
    )


# =========================================================
# TAB 3 — LIMITATIONS
# =========================================================

with tab3:


    st.markdown(
        """
        - Exact pixel-level segmentation is more difficult than image-level detection.
        - The defect classifiers are limited by the small number of real defect images.
        - Data augmentation does not create genuinely new physical defect mechanisms.
        - CAE performance is sensitive to alignment, texture variability and decoder capacity.
        - The router accuracy was measured on the available dataset and may decrease under new production conditions.
        - Production deployment still requires threshold calibration and latency benchmarking.
        """
    )


# =========================================================
# TAB 4 — NEXT STEPS
# =========================================================

with tab4:


    st.markdown(
        """
        1. Collect more independent real defect images.
        2. Calibrate category-specific thresholds on production data.
        3. Add repeated-seed evaluation for the CAE.
        4. Calculate CAE pixel-level metrics.
        5. Benchmark OpenVINO latency and memory usage.
        6. Add rejection logic for unknown object categories and unknown defects.
        7. Validate the pipeline with real industrial camera conditions.
        """
    )


# =========================================================
# FINAL TAKEAWAY
# =========================================================

show_takeaway(
    (
        "PatchCore achieved the strongest overall anomaly-detection "
        "performance and was therefore selected as the primary detector "
        "for the final inspection pipeline. The convolutional "
        "autoencoder remains a transparent reconstruction-based "
        "baseline that complements PatchCore through interpretability "
        "and comparative evaluation."
    )
)