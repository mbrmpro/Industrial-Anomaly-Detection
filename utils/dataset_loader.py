"""Backward-compatible API for the slim deployment dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from utils.config import DATASET_ROOT, DATASET_SAMPLES_PATH
from utils.data_loader import load_dataset_statistics

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _safe_name(value: str, label: str) -> str:
    name = str(value).strip()
    if not name:
        raise ValueError(f"{label} must not be empty.")
    if Path(name).name != name:
        raise ValueError(f"Invalid {label.lower()}: {value!r}")
    return name


def _image_paths(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _int_value(row: dict[str, Any], *keys: str, default: int = 0) -> int:
    for key in keys:
        if key not in row:
            continue
        value = row[key]
        if pd.isna(value):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return int(default)


@st.cache_data(show_spinner=False)
def get_categories() -> list[str]:
    """Return filesystem categories or fall back to dataset statistics."""
    if DATASET_ROOT.is_dir():
        categories = sorted(
            path.name for path in DATASET_ROOT.iterdir() if path.is_dir()
        )
        if categories:
            return categories

    dataframe = load_dataset_statistics()
    category_column = next(
        (
            column
            for column in ("category", "Category")
            if column in dataframe.columns
        ),
        None,
    )
    if category_column is None:
        return []

    return sorted(
        dataframe[category_column]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda values: values.ne("")]
        .unique()
        .tolist()
    )


@st.cache_data(show_spinner=False)
def get_statistics(category: str) -> dict[str, Any]:
    """Return CSV statistics with stable compatibility keys."""
    category_name = _safe_name(category, "Category")
    dataframe = load_dataset_statistics()

    if "category" not in dataframe.columns:
        raise KeyError("dataset_statistics.csv must contain a 'category' column.")

    rows = dataframe.loc[
        dataframe["category"].astype(str).str.strip() == category_name
    ]
    if rows.empty:
        raise KeyError(
            f"Category is not present in dataset statistics: {category_name}"
        )

    raw = rows.iloc[0].to_dict()
    train = _int_value(raw, "train", "train_images", "training_images")
    test_good = _int_value(
        raw, "test_good", "test_good_images", "good_test_images"
    )
    test_defect = _int_value(
        raw,
        "test_defect",
        "test_defect_images",
        "defect_test_images",
        "test_defective",
    )
    test = _int_value(
        raw, "test", "test_images", default=test_good + test_defect
    )
    if test == 0:
        test = test_good + test_defect
    total = _int_value(raw, "total", "total_images", default=train + test)
    if total == 0:
        total = train + test

    result = dict(raw)
    result.update(
        {
            "train": train,
            "test_good": test_good,
            "test_defect": test_defect,
            "test": test,
            "total": total,
        }
    )

    defect_names = result.get("defect_type_names")
    if isinstance(defect_names, str):
        result["defect_type_names"] = [
            item.strip() for item in defect_names.split("|") if item.strip()
        ]
    elif defect_names is not None and pd.isna(defect_names):
        result["defect_type_names"] = []

    return result


@st.cache_data(show_spinner=False)
def count_images(category: str) -> dict[str, int]:
    """Read precomputed counts from CSV; no filesystem counting."""
    stats = get_statistics(category)
    return {
        "train": int(stats["train"]),
        "test_good": int(stats["test_good"]),
        "test_defect": int(stats["test_defect"]),
        "test": int(stats["test"]),
        "total": int(stats["total"]),
    }


@st.cache_data(show_spinner=False)
def get_train_images(category: str) -> list[Path]:
    category_name = _safe_name(category, "Category")

    local_path = (
        DATASET_ROOT
        / category_name
        / "train"
        / "good"
    )

    local_images = _image_paths(local_path)
    if local_images:
        return local_images

    sample_path = (
        DATASET_SAMPLES_PATH
        / category_name
        / "train"
        / "good"
    )

    return _image_paths(sample_path)

@st.cache_data(show_spinner=False)
def get_test_defects(category: str) -> list[str]:
    category_name = _safe_name(category, "Category")

    candidate_roots = [
        DATASET_ROOT / category_name / "test",
        DATASET_SAMPLES_PATH / category_name / "test",
    ]

    for test_root in candidate_roots:
        if not test_root.is_dir():
            continue

        classes = sorted(
            path.name
            for path in test_root.iterdir()
            if path.is_dir()
        )

        if classes:
            if "good" in classes:
                return ["good"] + [
                    name for name in classes
                    if name != "good"
                ]
            return classes

    return []

@st.cache_data(show_spinner=False)
def get_defect_types(category: str) -> list[str]:
    return [name for name in get_test_defects(category) if name != "good"]


@st.cache_data(show_spinner=False)
def get_defect_type_names(category: str) -> list[str]:
    return get_defect_types(category)


@st.cache_data(show_spinner=False)
def get_test_images(
    category: str,
    defect: str,
) -> list[Path]:
    category_name = _safe_name(category, "Category")
    class_name = _safe_name(defect, "Class")

    candidate_paths = [
        DATASET_ROOT / category_name / "test" / class_name,
        DATASET_SAMPLES_PATH / category_name / "test" / class_name,
    ]

    for path in candidate_paths:
        images = _image_paths(path)
        if images:
            return images

    return []

@st.cache_data(show_spinner=False)
def get_test_good_images(category: str) -> list[Path]:
    return get_test_images(category, "good")


@st.cache_data(show_spinner=False)
def get_test_defect_images(category: str, defect_type: str) -> list[Path]:
    return get_test_images(category, defect_type)


@st.cache_data(show_spinner=False)
def get_ground_truth_images(
    category: str,
    defect: str,
) -> list[Path]:
    category_name = _safe_name(category, "Category")
    class_name = _safe_name(defect, "Class")

    if class_name == "good":
        return []

    candidate_paths = [
        DATASET_ROOT
        / category_name
        / "ground_truth"
        / class_name,

        DATASET_SAMPLES_PATH
        / category_name
        / "ground_truth"
        / class_name,
    ]

    for path in candidate_paths:
        images = _image_paths(path)
        if images:
            return images

    return []

@st.cache_data(show_spinner=False)
def get_ground_truth_masks(category: str, defect_type: str) -> list[Path]:
    return get_ground_truth_images(category, defect_type)