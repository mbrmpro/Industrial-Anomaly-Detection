import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from PIL import Image
import streamlit as st
import numpy as np
import torch

from components.kpi_cards import show_kpi_cards
from components.page_header import show_page_header
from components.takeaway import show_takeaway

from utils.config import APP_ROOT
from utils.inference_pipeline import (
    load_dinov2_model,
    run_inspection,
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
    title="Live Industrial Inspection",
    subtitle=(
        "Upload an industrial image and run the deployed "
        "object-routing, anomaly-detection and defect-classification pipeline."
    ),
    icon="🔎",
)


# =========================================================
# MODEL CACHE
# =========================================================

@st.cache_resource
def get_dinov2():
    """
    Load and cache the DINOv2 feature extractor.

    The model is loaded once per Streamlit session and reused for
    subsequent inspections.
    """

    return load_dinov2_model()


# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=[
        "png",
        "jpg",
        "jpeg",
    ],
)


if uploaded_file is None:

    st.info(
        "Upload a normal or defective MVTec AD image to start the inspection."
    )

    show_takeaway(
        (
            "The live workflow combines DINOv2 object routing, "
            "category-specific PatchCore anomaly detection, "
            "OpenVINO inference and category-specific defect classification."
        )
    )

    st.stop()


# =========================================================
# IMAGE PREVIEW
# =========================================================

try:

    image = Image.open(
        uploaded_file
    ).convert(
        "RGB"
    )

except Exception as error:

    st.error(
        "The uploaded file could not be opened as an RGB image."
    )

    st.exception(error)
    st.stop()


preview_col, details_col = st.columns(
    [1, 1.2]
)

with preview_col:

    st.image(
        image,
        caption="Uploaded image",
        width=300,
    )

with details_col:

    st.markdown("#### Input Information")

    input_width, input_height = image.size

    input_info = {
        "Filename": uploaded_file.name,
        "Resolution": f"{input_width} × {input_height}",
        "Colour Mode": image.mode,
        "File Type": uploaded_file.type or "Unknown",
    }

    for label, value in input_info.items():
        st.markdown(
            f"**{label}:** `{value}`"
        )

    st.caption(
        "The image will be resized and normalized internally "
        "according to the deployed inference pipeline."
    )


# =========================================================
# INSPECTION BUTTON
# =========================================================

run_button = st.button(
    "Run inspection",
    type="primary",
    use_container_width=True,
)


if not run_button:

    show_takeaway(
        (
            "The uploaded image is ready for inference. "
            "Run the inspection to obtain the object category, anomaly score, "
            "localization map and defect prediction."
        )
    )

    st.stop()


# =========================================================
# INFERENCE
# =========================================================

with st.spinner(
    "Running industrial inspection..."
):

    try:

        dino_model, device = get_dinov2()

        result = run_inspection(
            image=image,
            app_root=APP_ROOT,
            dino_model=dino_model,
            device=device,
        )

    except Exception as error:

        st.error(
            "The inspection pipeline could not complete the inference."
        )

        st.exception(error)
        st.stop()


# =========================================================
# RESULT VALIDATION
# =========================================================

required_result_keys = {
    "object_type",
    "is_anomaly",
    "anomaly_score",
    "defect_type",
}

missing_result_keys = (
    required_result_keys
    - set(result.keys())
)

if missing_result_keys:

    st.error(
        "Inference result is missing required values:\n\n"
        f"`{sorted(missing_result_keys)}`"
    )

    st.stop()


# =========================================================
# PREDICTION SUMMARY
# =========================================================

status_text = (
    "ANOMALY"
    if bool(result["is_anomaly"])
    else "NORMAL"
)

defect_text = (
    str(result["defect_type"])
    if result.get("defect_type")
    else "—"
)

show_kpi_cards(
    [
        (
            "Detected Object",
            str(result["object_type"]),
        ),
        (
            "Status",
            status_text,
        ),
        (
            "Anomaly Score",
            f"{float(result['anomaly_score']):.4f}",
        ),
        (
            "Predicted Defect",
            defect_text,
        ),
    ]
)


if bool(result["is_anomaly"]):

    st.error(
        "ANOMALY DETECTED"
    )

else:

    st.success(
        "NORMAL / GOOD COMPONENT"
    )


# =========================================================
# RESULT TABS
# =========================================================

result_tab, localization_tab, technical_tab = st.tabs(
    [
        "Inspection Result",
        "Anomaly Localization",
        "Technical Details",
    ]
)


# =========================================================
# TAB 1 — INSPECTION RESULT
# =========================================================

with result_tab:

    result_col, interpretation_col = st.columns(
        [1, 1]
    )

    with result_col:

        st.markdown("#### Prediction")

        st.markdown(
            f"""
            **Object category:** `{result["object_type"]}`  
            **Inspection status:** `{status_text}`  
            **Anomaly score:** `{float(result["anomaly_score"]):.4f}`  
            **Predicted defect:** `{defect_text}`
            """
        )

    with interpretation_col:

        st.markdown("#### Interpretation")

        if bool(result["is_anomaly"]):

            st.warning(
                """
                The category-specific PatchCore model classified the
                component as anomalous.

                The predicted defect label is produced by the
                category-specific polynomial SVM.
                """
            )

        else:

            st.info(
                """
                The category-specific PatchCore model classified the
                component as normal.

                The defect classifier is not required for a normal result.
                """
            )


# =========================================================
# TAB 2 — ANOMALY LOCALIZATION
# =========================================================

with localization_tab:

    heatmap = result.get(
        "heatmap"
    )

    overlay = result.get(
        "overlay"
    )

    if heatmap is None:

        st.info(
            "No anomaly heatmap is available for this inspection result."
        )

    else:

        original_col, heatmap_col, overlay_col = st.columns(
            3
        )

        with original_col:

            st.image(
                image,
                caption="Original image",
                use_container_width=True,
            )

        with heatmap_col:

            st.image(
                heatmap,
                caption="PatchCore heatmap",
                use_container_width=True,
            )

        with overlay_col:

            if overlay is not None:

                st.image(
                    overlay,
                    caption="Heatmap overlay",
                    use_container_width=True,
                )

            else:

                st.info(
                    "Overlay is not available."
                )

        st.caption(
            "The anomaly map is generated directly by the trained "
            "category-specific PatchCore OpenVINO model. "
            "The JET colour map and overlay are visualization steps only."
        )


# =========================================================
# TAB 3 — TECHNICAL DETAILS
# =========================================================

with technical_tab:

    technical_col, workflow_col = st.columns(
        [1, 1]
    )

    with technical_col:

        st.markdown("#### Deployed Components")

        st.markdown(
            f"""
            **Object feature extractor:** DINOv2 ViT-L/14  
            **Object router:** Logistic Regression  
            **Anomaly detector:** Category-specific PatchCore  
            **Inference engine:** OpenVINO  
            **Defect classifier:** Category-specific Polynomial SVM  
            **Execution device:** `{device}`
            """
        )

    with workflow_col:

        st.markdown("#### Conditional Workflow")

        st.code(
            """
Upload Image
    ↓
DINOv2 Object Router
    ↓
Category-Specific PatchCore
    ↓
Normal or Anomalous?
    ├── Normal → Final Result
    └── Anomalous
          ↓
      Heatmap + Defect SVM
          ↓
      Defect Type
            """,
            language="text",
        )


# =========================================================
# FINAL TAKEAWAY
# =========================================================

show_takeaway(
    (
        "The live inspection integrates the complete deployed pipeline: "
        "DINOv2 object routing, category-specific PatchCore anomaly detection, "
        "OpenVINO inference, spatial heatmap localization and conditional "
        "defect-type classification."
    )
)