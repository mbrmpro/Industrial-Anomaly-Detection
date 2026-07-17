import pandas as pd
import streamlit as st

from components.kpi_cards import show_kpi_cards
from components.page_header import show_page_header
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
    title="Feature Engineering with DINOv2",
    subtitle=(
        "Pretrained global and local image representations "
        "for object routing and defect-type classification."
    ),
    icon="🔬",
)


# =========================================================
# SUMMARY METRICS
# =========================================================

show_kpi_cards(
    [
        (
            "Backbone",
            "DINOv2 ViT-L/14",
        ),
        (
            "Input Resolution",
            "518 × 518",
        ),
        (
            "Patch Tokens",
            "1,369",
        ),
        (
            "Hybrid Feature",
            "2,048-D",
        ),
    ]
)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Role of DINOv2",
        "Input & Tokens",
        "Hybrid Feature",
        "Downstream Models",
    ]
)


# =========================================================
# TAB 1 — ROLE OF DINOV2
# =========================================================

with tab1:

    overview_col, distinction_col = st.columns(
        [1.1, 1]
    )

    with overview_col:

        st.markdown("#### Project Role")

        st.markdown(
            """
            DINOv2 is used as a fixed pretrained feature extractor.

            The transformer is not fine-tuned. Instead, its image
            representations are passed to classical machine-learning
            classifiers.
            """
        )

        st.markdown(
            """
            The extracted feature vector supports two tasks:

            - object-category routing;
            - category-specific defect classification.
            """
        )

    with distinction_col:

        st.markdown("#### Important Distinction")

        st.info(
            """
            DINOv2 is not the PatchCore backbone.

            PatchCore uses Anomalib's pretrained Wide-ResNet-50-2.
            DINOv2 is used for object routing and defect-type
            classification.
            """
        )

        st.code(
            """
Image
  ↓
DINOv2 Hybrid Feature
  ├── Logistic Regression
  │      └── Object Category
  └── Polynomial SVM
         └── Defect Type
            """,
            language="text",
        )


# =========================================================
# TAB 2 — INPUT AND TOKENS
# =========================================================

with tab2:

    transform_col, token_col = st.columns(
        [1.05, 1]
    )

    with transform_col:

        transformation_df = pd.DataFrame(
            {
                "Step": [
                    "RGB conversion",
                    "Resize",
                    "Tensor conversion",
                    "Normalization",
                ],
                "Configuration": [
                    "Three-channel RGB",
                    "518 × 518",
                    "PyTorch tensor",
                    "ImageNet mean and standard deviation",
                ],
            }
        )

        st.dataframe(
            transformation_df,
            use_container_width=True,
            hide_index=True,
            height=220,
        )

        st.code(
            """
T.Resize((518, 518))
T.ToTensor()
T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
            """,
            language="python",
        )

    with token_col:

        st.markdown("#### Transformer Tokens")

        st.metric(
            "Patch Grid",
            "37 × 37",
        )

        st.metric(
            "Patch Tokens",
            "1,369",
        )

        st.metric(
            "Token Dimension",
            "1,024",
        )

        st.caption(
            "518 is divisible by the 14-pixel DINOv2 patch size."
        )


# =========================================================
# TAB 3 — HYBRID FEATURE
# =========================================================

with tab3:

    feature_col, formula_col = st.columns(
        [1.05, 1]
    )

    with feature_col:

        feature_df = pd.DataFrame(
            {
                "Component": [
                    "CLS token",
                    "Mean patch token",
                    "Hybrid feature",
                ],
                "Dimension": [
                    1024,
                    1024,
                    2048,
                ],
                "Information": [
                    "Global semantic representation",
                    "Aggregated local visual representation",
                    "Combined global and local information",
                ],
            }
        )

        st.dataframe(
            feature_df,
            use_container_width=True,
            hide_index=True,
            height=200,
        )

    with formula_col:

        st.markdown("#### Feature Construction")

        st.latex(
            r"""
            z(x)=
            \left[
            c(x);
            \frac{1}{N_p}
            \sum_{j=1}^{N_p}
            p_j(x)
            \right]
            """
        )

        st.success(
            """
            The final 2,048-dimensional vector combines global
            object identity with aggregated local structure.
            """
        )


# =========================================================
# TAB 4 — DOWNSTREAM MODELS
# =========================================================

with tab4:

    router_tab, classifier_tab = st.tabs(
        [
            "Object Router",
            "Defect Classifier",
        ]
    )

    with router_tab:

        router_col, workflow_col = st.columns(
            [1.1, 1]
        )

        with router_col:

            router_df = pd.DataFrame(
                {
                    "Component": [
                        "Training images",
                        "Feature extractor",
                        "Classifier",
                        "Class balancing",
                        "Output classes",
                    ],
                    "Configuration": [
                        "Original train/good images",
                        "DINOv2 ViT-L/14",
                        "Logistic Regression",
                        "Balanced",
                        "15 MVTec AD categories",
                    ],
                }
            )

            st.dataframe(
                router_df,
                use_container_width=True,
                hide_index=True,
                height=250,
            )

        with workflow_col:

            st.code(
                """
Input Image
    ↓
DINOv2 Hybrid Feature
    ↓
Logistic Regression
    ↓
Object Category
    ↓
Load Matching PatchCore Model
                """,
                language="text",
            )

            st.info(
                """
                The reported 100% router accuracy applies to the
                available evaluation data and does not guarantee
                identical performance under new production conditions.
                """
            )

    with classifier_tab:

        classifier_col, limitation_col = st.columns(
            [1.1, 1]
        )

        with classifier_col:

            classifier_df = pd.DataFrame(
                {
                    "Element": [
                        "Classifier",
                        "Kernel",
                        "Degree",
                        "C",
                        "Class weight",
                        "Validation",
                        "Leakage control",
                    ],
                    "Configuration": [
                        "Support Vector Machine",
                        "Polynomial",
                        "5",
                        "100",
                        "Balanced",
                        "Stratified five-fold cross-validation",
                        "Augmentations only in the training fold",
                    ],
                }
            )

            st.dataframe(
                classifier_df,
                use_container_width=True,
                hide_index=True,
                height=300,
            )

        with limitation_col:

            st.warning(
                """
                Defect classes contain only a limited number of
                independent physical samples.

                Augmentation increases training volume but does not
                create genuinely new defect mechanisms.
                """
            )

            st.info(
                """
                Original images are split first. Augmented variants are
                added only when their original belongs to the training
                fold, preventing data leakage.
                """
            )


# =========================================================
# TAKEAWAY
# =========================================================

show_takeaway(
    (
        "DINOv2 provides one reusable 2,048-dimensional representation "
        "that combines global CLS information with averaged local patch "
        "features. This representation supports both object routing and "
        "category-specific defect classification."
    )
)