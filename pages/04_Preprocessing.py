from pathlib import Path

import pandas as pd
import streamlit as st

from components.kpi_cards import show_kpi_cards
from components.page_header import show_page_header
from components.takeaway import show_takeaway

from utils.config import APP_ROOT

from components.sidebar import show_sidebar
from utils.style_loader import load_css

# =========================================================
# GLOBAL APPLICATION ELEMENTS
# =========================================================
load_css()
show_sidebar()
# =========================================================
# PATHS
# =========================================================

PREPROCESSING_ASSETS = (
    APP_ROOT
    / "assets"
    / "preprocessing"
)

RESIZE_BEFORE_PATH = (
    PREPROCESSING_ASSETS
    / "resize_before.png"
)

RESIZE_AFTER_PATH = (
    PREPROCESSING_ASSETS
    / "resize_after.png"
)

NORMALIZATION_PATH = (
    PREPROCESSING_ASSETS
    / "normalization_before_after.png"
)

AUGMENTATION_EXAMPLES_PATH = (
    PREPROCESSING_ASSETS
    / "augmentation_examples.png"
)


# =========================================================
# HELPERS
# =========================================================


def show_optional_image(
    image_path: Path | str,
    caption: str | None = None,
    width: int | str | None = None,
) -> None:
    """
    Display an image when it exists.

    Width may be:
    - a positive integer in pixels
    - "stretch"
    - "content"
    - None, in which case the width argument is omitted
    """

    image_path = Path(image_path)

    if not image_path.exists():
        st.info(
            f"Image not available: {image_path.name}"
        )
        return

    image_kwargs = {
        "image": str(image_path),
    }

    if caption:
        image_kwargs["caption"] = caption

    if width is not None:
        image_kwargs["width"] = width

    st.image(
        **image_kwargs
    )

# =========================================================
# DATA
# =========================================================

augmentation_counts = pd.DataFrame(
    {
        "Category": [
            "bottle",
            "cable",
            "capsule",
            "carpet",
            "grid",
            "hazelnut",
            "leather",
            "metal_nut",
            "pill",
            "screw",
            "tile",
            "transistor",
            "wood",
            "zipper",
        ],
        "Original": [
            209,
            224,
            219,
            280,
            264,
            391,
            245,
            220,
            267,
            320,
            230,
            213,
            247,
            240,
        ],
        "After Augmentation": [
            1254,
            1344,
            1314,
            1680,
            1584,
            2346,
            1470,
            1320,
            1602,
            1920,
            1380,
            1278,
            1482,
            1440,
        ],
    }
)

augmentation_counts["Factor"] = (
    augmentation_counts["After Augmentation"]
    / augmentation_counts["Original"]
).round(1)

original_total = int(
    augmentation_counts["Original"].sum()
)

augmented_total = int(
    augmentation_counts["After Augmentation"].sum()
)

growth_percentage = (
    (augmented_total - original_total)
    / original_total
    * 100
)


# =========================================================
# PAGE HEADER
# =========================================================

show_page_header(
    title="Preprocessing and Data Augmentation",
    subtitle=(
        "Image standardization and object-safe offline augmentation "
        "for reproducible model training."
    ),
    icon="⚙️",
)


show_kpi_cards(
    [
        (
            "Target Resolution",
            "256 × 256",
        ),
        (
            "Normalized Range",
            "[0, 1]",
        ),
        (
            "Original Images",
            f"{original_total:,}",
        ),
        (
            "After Augmentation",
            f"{augmented_total:,}",
        ),
    ]
)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Pipeline",
        "Resize & Normalize",
        "Augmentation",
        "Dataset Growth",
    ]
)


# =========================================================
# TAB 1 — PIPELINE
# =========================================================

with tab1:

    st.markdown(
        """
        The preprocessing pipeline creates a consistent and reproducible
        input representation before model training.
        """
    )

    st.code(
        """
Input Image
    ↓
Resize to 256 × 256
    ↓
Normalize to [0, 1]
    ↓
Convert to Tensor
    ↓
Offline Data Augmentation
    ↓
DataLoader
    ↓
Model Training
        """,
        language="text",
    )

    pipeline_table = pd.DataFrame(
        {
            "Step": [
                "Resize",
                "Normalize",
                "Tensor Conversion",
                "Offline Augmentation",
                "DataLoader",
            ],
            "Purpose": [
                "Create identical spatial dimensions",
                "Improve numerical stability",
                "Enable PyTorch computation",
                "Increase realistic training variability",
                "Organize images into mini-batches",
            ],
        }
    )

    st.dataframe(
        pipeline_table,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Offline augmentation reduces computation during training "
        "and ensures reproducible experiments."
    )


# =========================================================
# TAB 2 — RESIZE AND NORMALIZATION
# =========================================================

with tab2:

    image_col1, image_col2, formula_col = st.columns(
        [1, 1, 0.9]
    )

    with image_col1:
        st.markdown("#### Before resize")

        show_optional_image(
            RESIZE_BEFORE_PATH,
            "Original image",
            width=220,
        )

    with image_col2:
        st.markdown("#### After resize")

        show_optional_image(
            RESIZE_AFTER_PATH,
            "256 × 256 image",
            width=220,
        )

    with formula_col:
        st.markdown("#### Normalization")

        st.latex(
            r"x_{\mathrm{norm}}=\frac{x}{255}"
        )

        st.metric(
            "Before",
            "0–255",
        )

        st.metric(
            "After",
            "0–1",
        )

    if NORMALIZATION_PATH.exists():

        with st.expander(
            "Show normalization distribution",
            expanded=False,
        ):
            show_optional_image(
                NORMALIZATION_PATH,
                "Pixel distribution before and after normalization",
            )

    st.success(
        "Resizing enables batch processing, while normalization "
        "improves numerical stability during optimization."
    )


# =========================================================
# TAB 3 — DATA AUGMENTATION
# =========================================================

with tab3:

    st.markdown(
        """
        Object-safe transformations are applied only to normal training
        images. Transformations remain small to preserve the physical
        geometry of each industrial component.
        """
    )

    augmentation_table = pd.DataFrame(
        {
            "Transformation": [
                "Horizontal / vertical flip",
                "Small rotation",
                "Micro-translation",
                "Random zoom",
                "Brightness / contrast",
                "Gaussian noise",
                "Gaussian blur",
            ],
            "Simulated Variation": [
                "Orientation",
                "Camera angle",
                "Positioning inaccuracy",
                "Camera distance",
                "Illumination",
                "Sensor noise",
                "Slight focus degradation",
            ],
        }
    )

    table_col, image_col = st.columns(
        [1.05, 1]
    )

    with table_col:
        st.dataframe(
            augmentation_table,
            use_container_width=True,
            hide_index=True,
        )

    with image_col:
        show_optional_image(
            AUGMENTATION_EXAMPLES_PATH,
            "Original and augmented examples",
        )

    st.warning(
        "The final numerical augmentation parameters remain pending "
        "until confirmation by the project team."
    )

    st.info(
        "Test images are not augmented. This preserves an unbiased "
        "evaluation on the original test distribution."
    )


# =========================================================
# TAB 4 — DATASET GROWTH
# =========================================================

with tab4:

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Original Images",
        f"{original_total:,}",
    )

    metric2.metric(
        "After Augmentation",
        f"{augmented_total:,}",
    )

    metric3.metric(
        "Dataset Growth",
        f"{growth_percentage:.0f}%",
    )

    st.dataframe(
        augmentation_counts,
        use_container_width=True,
        hide_index=True,
        height=330,
    )

    st.success(
        "Each original image generated five additional samples, "
        "resulting in an effective augmentation factor of 6×."
    )

    st.caption(
        "The toothbrush category is excluded from this augmentation "
        "summary according to the project preprocessing decision."
    )


# =========================================================
# TAKEAWAY
# =========================================================

show_takeaway(
    (
        "The preprocessing pipeline standardizes all model inputs and "
        "increases normal training-data variability through reproducible, "
        "object-safe offline augmentation."
    )
)