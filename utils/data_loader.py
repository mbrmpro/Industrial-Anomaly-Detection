"""Central cached loaders for deployment assets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from utils.config import (
    CAE_METRICS_PATH,
    DATASET_STATISTICS_PATH,
    DEFECT_CLASSIFIER_METRICS_PATH,
    DEFECT_STATISTICS_PATH,
    PATCHCORE_METRICS_PATH,
    RGB_DENSITY_PATH,
)


@st.cache_data(show_spinner=False)
def load_csv(path: str | Path) -> pd.DataFrame:
    """Load and cache a CSV file."""
    csv_path = Path(path).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    try:
        return pd.read_csv(csv_path)
    except Exception as error:
        raise RuntimeError(f"Could not read CSV file: {csv_path}") from error


@st.cache_data(show_spinner=False)
def load_npz(path: str | Path) -> dict[str, Any]:
    """Load and cache an NPZ file as a normal dictionary."""
    npz_path = Path(path).expanduser()
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}
    except Exception as error:
        raise RuntimeError(f"Could not read NPZ file: {npz_path}") from error


@st.cache_data(show_spinner=False)
def load_dataset_statistics() -> pd.DataFrame:
    return load_csv(DATASET_STATISTICS_PATH)


@st.cache_data(show_spinner=False)
def load_defect_statistics() -> pd.DataFrame:
    return load_csv(DEFECT_STATISTICS_PATH)


@st.cache_data(show_spinner=False)
def load_rgb_density() -> dict[str, Any]:
    return load_npz(RGB_DENSITY_PATH)


@st.cache_data(show_spinner=False)
def load_patchcore_metrics() -> pd.DataFrame:
    return load_csv(PATCHCORE_METRICS_PATH)


@st.cache_data(show_spinner=False)
def load_defect_classifier_metrics() -> pd.DataFrame:
    return load_csv(DEFECT_CLASSIFIER_METRICS_PATH)


@st.cache_data(show_spinner=False)
def load_cae_metrics() -> pd.DataFrame:
    return load_csv(CAE_METRICS_PATH)
