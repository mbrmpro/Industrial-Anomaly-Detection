"""
=========================================================
Industrial Anomaly Detection
Global Configuration
=========================================================
"""

from pathlib import Path

# =========================================================
# PROJECT
# =========================================================
PROJECT_NAME = "Industrial Anomaly Detection"
PROJECT_SUBTITLE = (
    "Anomaly Detection in Industrial Components "
    "using Deep Learning"
)
VERSION = "1.0"
AUTHOR = "Ayoub,Christopher,Mohamed,Romeal"
# =========================================================

# =========================================================
# DATASET INFORMATION
# =========================================================
DATASET_NAME = "MVTec AD"
DEFAULT_CATEGORY = "Bottle"
TOTAL_CATEGORIES = 14
TOTAL_IMAGES = 21414
# =========================================================
# PROJECT INFORMATION
# =========================================================
PROJECT_TYPE = "Industrial Computer Vision"
TASK = "Image Anomaly Detection"
FRAMEWORK = "TensorFlow + PyTorch"

# ML_PROJECT = Path(
#     r"D:\@@Liora_ML_Engineer_2026\10_Project\Anomaly_detection"
# )
# DATASET_PATH = ML_PROJECT / "Dataset_preprocessed"
# MODELS_PATH = ML_PROJECT / "Models"
# RESULTS_PATH = ML_PROJECT / "Results"

# # =========================================================
# # STREAMLIT APP
# # =========================================================

# APP_ROOT = Path(__file__).resolve().parent.parent

# ASSETS_PATH = APP_ROOT / "assets"

# EDA_ASSETS_PATH = ASSETS_PATH / "eda"

# #RGB_CACHE_PATH = EDA_ASSETS_PATH / "rgb_distribution_data.npz"
# RGB_CACHE_PATH = (
#     APP_ROOT
#     / "assets"
#     / "eda"
#     / "rgb_distribution_data.npz"
# )

# RGB_DENSITY_PATH = (
#     APP_ROOT
#     / "assets"
#     / "eda"
#     / "rgb_density_data.npz"
# )
# # =========================================================
# # MODELS
# # =========================================================
# NUMBER_OF_MODELS = 4


# =========================================================
# LOCAL MACHINE LEARNING PROJECT
# =========================================================

ML_PROJECT = Path(
    r"D:\@@Liora_ML_Engineer_2026\10_Project\Anomaly_detection"
)

LOCAL_DATASET_PATH = (
    ML_PROJECT
    / "Dataset_preprocessed"
)

MODELS_PATH = ML_PROJECT / "Models"
RESULTS_PATH = ML_PROJECT / "Results"


# =========================================================
# STREAMLIT APPLICATION
# =========================================================

APP_ROOT = Path(__file__).resolve().parent.parent

ASSETS_PATH = APP_ROOT / "assets"

EDA_ASSETS_PATH = ASSETS_PATH / "eda"
PREPROCESSING_ASSETS_PATH = ASSETS_PATH / "preprocessing"
PATCHCORE_ASSETS_PATH = ASSETS_PATH / "patchcore"


DATASET_SUMMARY_PATH = (
    EDA_ASSETS_PATH
    / "dataset_summary.csv"
)


DEFECT_STATISTICS_PATH = (
    EDA_ASSETS_PATH
    / "defect_statistics.csv"
)

RGB_DENSITY_PATH = (
    EDA_ASSETS_PATH
    / "rgb_density_data.npz"
)



PATCHCORE_METRICS_PATH = (
    PATCHCORE_ASSETS_PATH
    / "patchcore_metrics.csv"
)


# Backward compatibility for existing pages
DATASET_PATH = LOCAL_DATASET_PATH


# =========================================================
# CAE
# =========================================================

CAE_ASSETS_PATH = (
    ASSETS_PATH
    / "cae"
)

CAE_METRICS_PATH = (
    CAE_ASSETS_PATH
    / "cae_metrics.csv"
)

PATCHCORE_ASSETS_PATH = (
    ASSETS_PATH
    / "patchcore"
)

PATCHCORE_METRICS_PATH = (
    PATCHCORE_ASSETS_PATH
    / "patchcore_metrics.csv"
)

DEFECT_CLASSIFIER_METRICS_PATH = (
    PATCHCORE_ASSETS_PATH
    / "defect_classifier_metrics.csv"
)


# =========================================================
# ORIGINAL PROJECT METRICS
# =========================================================

METRICS_ASSETS_PATH = (
    ASSETS_PATH
    / "metrics"
)

DEFECT_CLASSIFIER_REPORTS_PATH = (
    METRICS_ASSETS_PATH
    / "defect_clf"
)

# =========================================================
# PRESENTATION FIGURES
# =========================================================

# =========================================================
# PRESENTATION FIGURE SIZE
# =========================================================
FIGURE_WIDTH = 8.5
FIGURE_HEIGHT = 4
FIGURE_DPI = 110