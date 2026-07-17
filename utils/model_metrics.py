from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


REPORT_PATTERN = re.compile(
    r"defect_clf_avg_report_(.+)_svm\.json$"
)


def _extract_mean_std(
    metric_entry: object,
) -> tuple[float | None, float | None]:
    """
    Read metric values stored as:

    {
        "mean": 0.92,
        "std": 0.08,
        "n": 5
    }
    """

    if not isinstance(metric_entry, dict):
        return None, None

    mean_value = metric_entry.get("mean")
    std_value = metric_entry.get("std")

    mean_value = (
        float(mean_value)
        if mean_value is not None
        else None
    )

    std_value = (
        float(std_value)
        if std_value is not None
        else None
    )

    return mean_value, std_value


def load_defect_classifier_reports(
    reports_dir: str | Path,
) -> pd.DataFrame:
    """
    Load the original averaged classification-report JSON files.

    Returns one row per object category.
    """

    reports_dir = Path(reports_dir)

    if not reports_dir.exists():
        raise FileNotFoundError(
            f"Defect-classifier reports not found: "
            f"{reports_dir}"
        )

    records: list[dict] = []

    report_paths = sorted(
        reports_dir.glob(
            "defect_clf_avg_report_*_svm.json"
        )
    )

    if not report_paths:
        raise FileNotFoundError(
            "No defect-classifier JSON reports found in "
            f"{reports_dir}"
        )

    for report_path in report_paths:

        match = REPORT_PATTERN.match(
            report_path.name
        )

        if match is None:
            continue

        category = match.group(1)

        with report_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            report = json.load(file)

        macro_average = report.get(
            "macro avg",
            {}
        )

        weighted_average = report.get(
            "weighted avg",
            {}
        )

        macro_precision, macro_precision_std = (
            _extract_mean_std(
                macro_average.get("precision")
            )
        )

        macro_recall, macro_recall_std = (
            _extract_mean_std(
                macro_average.get("recall")
            )
        )

        macro_f1, macro_f1_std = (
            _extract_mean_std(
                macro_average.get("f1-score")
            )
        )

        weighted_f1, weighted_f1_std = (
            _extract_mean_std(
                weighted_average.get("f1-score")
            )
        )

        accuracy, accuracy_std = (
            _extract_mean_std(
                report.get("accuracy")
            )
        )

        records.append(
            {
                "category": category,
                "accuracy_mean": accuracy,
                "accuracy_std": accuracy_std,
                "macro_precision_mean":
                    macro_precision,
                "macro_precision_std":
                    macro_precision_std,
                "macro_recall_mean":
                    macro_recall,
                "macro_recall_std":
                    macro_recall_std,
                "macro_f1_mean":
                    macro_f1,
                "macro_f1_std":
                    macro_f1_std,
                "weighted_f1_mean":
                    weighted_f1,
                "weighted_f1_std":
                    weighted_f1_std,
                "report_file":
                    report_path.name,
            }
        )

    dataframe = pd.DataFrame(records)

    if dataframe.empty:
        raise RuntimeError(
            "The JSON reports were found, but no valid "
            "metrics could be extracted."
        )

    return dataframe.sort_values(
        "category"
    ).reset_index(drop=True)