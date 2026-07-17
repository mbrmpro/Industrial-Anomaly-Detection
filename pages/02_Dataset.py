import streamlit as st

from components.kpi_cards import show_kpi_cards
from components.page_header import show_page_header
from components.takeaway import show_takeaway

from utils.dataset_loader import (
    count_images,
    get_categories,
    get_test_defects,
    get_test_images,
)

from utils.image_viewer import (
    load_images,
    sample_images,
)

from components.sidebar import show_sidebar
from utils.style_loader import load_css








# =========================================================
# GLOBAL APPLICATION ELEMENTS
# =========================================================
load_css()
show_sidebar()
#show_header()

# =========================================================
# PAGE HEADER
# =========================================================

show_page_header(
    title="Dataset",
    subtitle=(
        "MVTec AD categories, train/test structure, "
        "normal images, defect types and ground-truth masks."
    ),
    icon="📦",
)


# =========================================================
# DATASET OVERVIEW
# =========================================================

categories = get_categories()
if not categories:
    st.error(
        "No dataset categories or cloud sample folders were found."
    )
    st.stop()
show_kpi_cards(
    [
        (
            "Categories",
            str(len(categories)),
        ),
        (
            "Training Setting",
            "One-Class",
        ),
        (
            "Training Images",
            "Normal Only",
        ),
        (
            "Ground Truth",
            "Pixel Masks",
        ),
    ]
)


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Overview",
        "Category Statistics",
        "Defect Types",
        "Visual Samples",
    ]
)


# =========================================================
# TAB 1 — OVERVIEW
# =========================================================

with tab1:

    st.subheader("MVTec AD Dataset")

    st.markdown(
        """
        The MVTec AD dataset is a benchmark dataset for industrial
        anomaly detection.

        It contains 15 industrial object and texture categories.
        Training data consists of normal images, while the test split
        contains both normal and defective samples.

        Pixel-level ground-truth masks are available for defective
        test images and support anomaly-localization evaluation.
        """
    )

    st.markdown("### Dataset Structure")

    st.code(
        """
Dataset_preprocessed/

└── <category>

      ├── train
      │      └── good
      │
      ├── test
      │      ├── good
      │      └── <defect_type>
      │
      └── ground_truth
             └── <defect_type>
        """,
        language="text",
    )

    st.info(
        """
        Training uses normal images only. Defective images and
        ground-truth masks are used during testing and evaluation.
        """
    )


# =========================================================
# TAB 2 — CATEGORY STATISTICS
# =========================================================

with tab2:

    st.subheader("Category Statistics")

    selected_category = st.selectbox(
        "Select category",
        categories,
        index=(
            categories.index("bottle")
            if "bottle" in categories
            else 0
        ),
        key="dataset_category_statistics",
    )

    stats = count_images(
        selected_category
    )

    stat1, stat2, stat3 = st.columns(3)

    stat1.metric(
        "Train Images",
        stats["train"],
    )

    stat2.metric(
        "Test Images",
        stats["test"],
    )

    stat3.metric(
        "Total Images",
        stats["total"],
    )

    st.markdown(
        f"""
        The selected category is **{selected_category}**.

        Its training split contains normal images only.
        The test split contains normal images and one or more
        category-specific defect types.
        """
    )


# =========================================================
# TAB 3 — DEFECT TYPES
# =========================================================

with tab3:

    st.subheader("Category-Specific Defect Types")

    selected_category_defects = st.selectbox(
        "Select category",
        categories,
        index=(
            categories.index("bottle")
            if "bottle" in categories
            else 0
        ),
        key="dataset_category_defects",
    )

    defects = get_test_defects(
        selected_category_defects
    )

    if defects:

        selected_defect = st.selectbox(
            "Select defect type",
            defects,
            key="dataset_defect_type",
        )

        st.markdown(
            f"""
            **Category:** `{selected_category_defects}`  
            **Selected test class:** `{selected_defect}`
            """
        )

        if selected_defect == "good":

            st.success(
                "The selected test class contains normal samples."
            )

        else:

            st.warning(
                """
                The selected test class contains defective samples.
                Ground-truth masks are available for localization
                evaluation.
                """
            )

    else:

        st.info(
            "No test defect directories were found for this category."
        )


# =========================================================
# TAB 4 — VISUAL SAMPLES
# =========================================================

with tab4:

    st.subheader("Representative Test Images")

    selected_category_samples = st.selectbox(
        "Select category",
        categories,
        index=(
            categories.index("bottle")
            if "bottle" in categories
            else 0
        ),
        key="dataset_category_samples",
    )

    sample_defects = get_test_defects(
        selected_category_samples
    )

    if not sample_defects:

        st.info(
            "No test classes were found for this category."
        )

    else:

        selected_sample_class = st.selectbox(
            "Select test class",
            sample_defects,
            key="dataset_sample_class",
        )

        image_paths = get_test_images(
            selected_category_samples,
            selected_sample_class,
        )
        #
        if not image_paths:

            st.info(
                "No prepared sample images are available "
                "for this category and test class."
            )

        else:

            sample = sample_images(
                image_paths,
                n=4,
            )

            images = load_images(
                sample
            )

            columns = st.columns(4)

            for index, image in enumerate(images):

                with columns[index % 4]:

                    st.image(
                        image,
                        width=150,
                    )

        #
        sample = sample_images(
            image_paths,
            n=4,
        )

        images = load_images(
            sample
        )

        if not images:

            st.info(
                "No sample images were found."
            )

        else:

            columns = st.columns(4)

            for index, image in enumerate(images):

                with columns[index]:

                    st.image(
                        image,
                        width=150,
                        caption=f"Sample {index+1}",
                        
                    )


# =========================================================
# TAKEAWAY
# =========================================================

show_takeaway(
    (
        "MVTec AD supports a realistic one-class anomaly-detection "
        "setting: models learn normal appearance from the training split "
        "and are evaluated using normal and defective test images together "
        "with pixel-level ground-truth masks."
    )
)