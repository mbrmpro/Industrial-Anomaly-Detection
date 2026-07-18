"""
=========================================================
Industrial Anomaly Detection
Global Configuration
=========================================================
"""


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

from pathlib import Path
APP_ROOT = Path(__file__).resolve().parent.parent
# =========================================================
# 
# =========================================================

ML_PROJECT = Path(
    r"D:\@@Liora_ML_Engineer_2026\10_Project\Anomaly_detection"
)

LOCAL_DATASET_PATH = ML_PROJECT / "Dataset_preprocessed"

CLOUD_DATASET_PATH = APP_ROOT / "assets" / "dataset"

if CLOUD_DATASET_PATH.is_dir():
    DATASET_ROOT = CLOUD_DATASET_PATH
elif LOCAL_DATASET_PATH.is_dir():
    DATASET_ROOT = LOCAL_DATASET_PATH
else:
    # Stable fallback for slim deployment
    DATASET_ROOT = CLOUD_DATASET_PATH

DATASET_PATH = DATASET_ROOT


ASSETS_PATH = APP_ROOT / "assets"
EDA_ASSETS_PATH = ASSETS_PATH / "eda"

MODELS_PATH = APP_ROOT / "Models"
RESULTS_PATH = APP_ROOT / "Results"


DATASET_SUMMARY_PATH = EDA_ASSETS_PATH / "dataset_summary.csv"
# Backward compatibility
DEFECT_STATISTICS_PATH = EDA_ASSETS_PATH / "defect_statistics.csv"
RGB_DENSITY_PATH = EDA_ASSETS_PATH / "rgb_density_data.npz"
###

DATASET_STATISTICS_PATH = EDA_ASSETS_PATH / "dataset_statistics.csv"

####
PREPROCESSING_ASSETS_PATH = ASSETS_PATH / "preprocessing"
PATCHCORE_ASSETS_PATH = ASSETS_PATH / "patchcore"
CAE_ASSETS_PATH = ASSETS_PATH / "cae"
METRICS_ASSETS_PATH = ASSETS_PATH / "metrics"



PATCHCORE_METRICS_PATH = (
    PATCHCORE_ASSETS_PATH / "patchcore_metrics.csv"
)

DEFECT_CLASSIFIER_METRICS_PATH = (
    PATCHCORE_ASSETS_PATH
    / "defect_classifier_metrics.csv"
)

CAE_METRICS_PATH = CAE_ASSETS_PATH / "cae_metrics.csv"

DEFECT_CLASSIFIER_REPORTS_PATH = (
    METRICS_ASSETS_PATH
    / "defect_clf"
)


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
# PRESENTATION FIGURE SIZE
# =========================================================
FIGURE_WIDTH = 8.5
FIGURE_HEIGHT = 4
FIGURE_DPI = 110