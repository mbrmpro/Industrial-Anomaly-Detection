import matplotlib.pyplot as plt
import streamlit as st

from components.kpi_cards import show_kpi_cards
from components.page_header import show_page_header

from utils.config import (
    DATASET_PATH,
    DEFECT_STATISTICS_PATH,
    RGB_DENSITY_PATH,
)

from utils.eda import (
    plot_defect_area_distribution,
    plot_defect_severity,
    plot_defect_types_per_category,
    plot_good_vs_defect_distribution,
    plot_rgb_intensity_distribution,
    plot_total_images_per_category,
    scan_mvtec_dataset,
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
    title="Exploratory Data Analysis",
    subtitle=(
        "Image distribution, defect diversity, defect severity "
        "and RGB intensity analysis across MVTec AD."
    ),
    icon="📊",
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_dataset_summary(path):
    return scan_mvtec_dataset(path)


try:
    df = load_dataset_summary(
        DATASET_PATH
    )

except Exception as error:
    st.exception(error)
    st.stop()


# =========================================================
# SUMMARY METRICS
# =========================================================

total_images = int(
    df["Images"].sum()
)

categories = int(
    df["Category"].nunique()
)

normal_images = int(
    df.loc[
        df["Class"] == "good",
        "Images",
    ].sum()
)

defective_images = int(
    df.loc[
        df["Class"] != "good",
        "Images",
    ].sum()
)


show_kpi_cards(
    [
        (
            "Images",
            f"{total_images:,}",
        ),
        (
            "Categories",
            str(categories),
        ),
        (
            "Normal Images",
            f"{normal_images:,}",
        ),
        (
            "Defective Images",
            f"{defective_images:,}",
        ),
    ]
)


# =========================================================
# PRESENTATION TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Image Counts",
        "Good vs Defective",
        "Defect Diversity",
        "Defect Severity",
        "RGB Analysis",
    ]
)


# =========================================================
# TAB 1 — IMAGE COUNTS
# =========================================================

with tab1:

    st.caption(
        "The number of available images differs between categories, "
        "which introduces category-level imbalance."
    )

    figure = plot_total_images_per_category(
        df
    )

    st.pyplot(
        figure,
        use_container_width=True,
    )

    plt.close(figure)


# =========================================================
# TAB 2 — GOOD VS DEFECTIVE
# =========================================================

with tab2:

    st.caption(
        "Good images are more frequent than defective images "
        "in every category."
    )

    figure = plot_good_vs_defect_distribution(
        df
    )

    st.pyplot(
        figure,
        use_container_width=True,
    )

    plt.close(figure)


# =========================================================
# TAB 3 — DEFECT DIVERSITY
# =========================================================

with tab3:

    st.caption(
        "The number of available defect types varies strongly "
        "between MVTec AD categories."
    )

    figure = plot_defect_types_per_category(
        df
    )

    st.pyplot(
        figure,
        use_container_width=True,
    )

    plt.close(figure)


# =========================================================
# TAB 4 — DEFECT SEVERITY
# =========================================================

with tab4:

    severity_tab, area_tab = st.tabs(
        [
            "Severity by Category",
            "Area Distribution",
        ]
    )

    with severity_tab:

        st.caption(
            "Ground-truth masks show that defect severity "
            "varies considerably between categories."
        )

        figure = plot_defect_severity(
            DEFECT_STATISTICS_PATH
        )

        st.pyplot(
            figure,
            use_container_width=True,
        )

        plt.close(figure)

    with area_tab:

        st.caption(
            "Defect areas span a broad range, from very small "
            "local anomalies to large damaged regions."
        )

        figure = plot_defect_area_distribution(
            DEFECT_STATISTICS_PATH
        )

        st.pyplot(
            figure,
            use_container_width=True,
        )

        plt.close(figure)

        st.info(
            """
            The logarithmic scale reveals both very small and very
            large defects. Each value is calculated from a real
            ground-truth mask.
            """
        )


# =========================================================
# TAB 5 — RGB ANALYSIS
# =========================================================

with tab5:

    st.caption(
        "Solid curves represent normal images; dashed curves "
        "represent anomalous images."
    )

    figure = plot_rgb_intensity_distribution(
        RGB_DENSITY_PATH
    )

    st.pyplot(
        figure,
        use_container_width=True,
    )

    plt.close(figure)

    st.info(
        """
        Normal and anomalous RGB distributions overlap strongly.
        Pixel intensity alone cannot reliably separate normal and
        defective samples. The anomaly signal depends more strongly
        on spatial structure, texture and local consistency.
        """
    )


# =========================================================
# COMPACT CONCLUSION
# =========================================================

with st.expander(
    "EDA Conclusion",
    expanded=False,
):

    st.markdown(
        """
        - The dataset is imbalanced across categories.
        - Training contains normal images only.
        - Good images are more frequent than defective images.
        - Defect type diversity differs strongly between categories.
        - Defect size and severity vary considerably.
        - RGB distributions overlap strongly.
        """
    )

    st.success(
        """
        These findings motivate deep feature-based, patch-level
        anomaly detection methods that preserve spatial information.
        """
    )