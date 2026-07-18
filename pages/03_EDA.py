from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from components.kpi_cards import show_kpi_cards
from components.page_header import show_page_header
from components.sidebar import show_sidebar
from utils.config import DATASET_PATH, DEFECT_STATISTICS_PATH, RGB_DENSITY_PATH
from utils.data_loader import load_dataset_statistics
from utils.eda import (
    plot_defect_area_distribution,
    plot_defect_severity,
    plot_defect_types_per_category,
    plot_good_vs_defect_distribution,
    plot_rgb_intensity_distribution,
    plot_total_images_per_category,
    scan_mvtec_dataset,
)
from utils.style_loader import load_css

load_css()
show_sidebar()

show_page_header(
    title="Exploratory Data Analysis",
    subtitle=(
        "Image distribution, defect diversity, defect severity "
        "and RGB intensity analysis across MVTec AD."
    ),
    icon="📊",
)


def _as_int(row: pd.Series, *keys: str, default: int = 0) -> int:
    for key in keys:
        if key not in row.index:
            continue
        value = row[key]
        if pd.isna(value):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return int(default)


def _defect_names(row: pd.Series) -> list[str]:
    value = row.get("defect_type_names")

    if isinstance(value, str):
        separator = "|" if "|" in value else ","
        return [
            item.strip()
            for item in value.split(separator)
            if item.strip() and item.strip().lower() != "good"
        ]

    if isinstance(value, (list, tuple, set)):
        return [
            str(item).strip()
            for item in value
            if str(item).strip() and str(item).strip().lower() != "good"
        ]

    count = _as_int(
        row,
        "defect_types",
        "defect_type_count",
        "n_defect_types",
        default=0,
    )
    return [f"defect_{index + 1}" for index in range(count)]


def _statistics_to_eda_frame(statistics: pd.DataFrame) -> pd.DataFrame:
    """Convert compact deployment statistics to Category/Class/Images rows."""
    if statistics.empty:
        return pd.DataFrame(columns=["Category", "Dataset", "Class", "Images"])

    category_column = next(
        (column for column in ("category", "Category") if column in statistics.columns),
        None,
    )
    if category_column is None:
        raise KeyError("dataset_statistics.csv must contain a 'category' column.")

    records: list[dict[str, object]] = []

    for _, row in statistics.iterrows():
        category = str(row[category_column]).strip()
        if not category:
            continue

        train = _as_int(row, "train", "train_images", "training_images")
        test_good = _as_int(
            row,
            "test_good",
            "test_good_images",
            "good_test_images",
        )
        defect_total = _as_int(
            row,
            "test_defect",
            "test_defect_images",
            "defect_test_images",
            "test_defective",
        )

        records.append(
            {
                "Category": category,
                "Dataset": "train",
                "Class": "good",
                "Images": train,
            }
        )

        records.append(
            {
                "Category": category,
                "Dataset": "test",
                "Class": "good",
                "Images": test_good,
            }
        )

        names = _defect_names(row)
        if not names and defect_total > 0:
            names = ["defect"]

        if names:
            base_count, remainder = divmod(defect_total, len(names))
            for index, name in enumerate(names):
                records.append(
                    {
                        "Category": category,
                        "Dataset": "test",
                        "Class": name,
                        "Images": base_count + (1 if index < remainder else 0),
                    }
                )

    return pd.DataFrame.from_records(
        records,
        columns=["Category", "Dataset", "Class", "Images"],
    )


@st.cache_data(show_spinner=False)
def load_dataset_summary(path: str) -> tuple[pd.DataFrame, str]:
    """Use the real dataset locally and CSV statistics on Streamlit Cloud."""
    dataset_path = Path(path)

    if dataset_path.is_dir():
        dataframe = scan_mvtec_dataset(dataset_path)
        if not dataframe.empty:
            return dataframe, f"Filesystem scan: {dataset_path}"

    statistics = load_dataset_statistics()
    dataframe = _statistics_to_eda_frame(statistics)

    if dataframe.empty:
        raise FileNotFoundError(
            "Neither a readable dataset directory nor usable "
            "dataset_statistics.csv data was found."
        )

    return dataframe, "Precomputed dataset statistics"


def _show_optional_plot(path: Path, plot_function, missing_message: str) -> None:
    if not path.is_file():
        st.warning(missing_message)
        return

    try:
        figure = plot_function(path)
        st.pyplot(figure, width="stretch")
        plt.close(figure)
    except Exception as error:
        st.warning(f"Analysis could not be displayed: {error}")


try:
    df, data_source = load_dataset_summary(str(DATASET_PATH))
except Exception as error:
    st.error(
        "EDA data could not be loaded. Check the deployed statistics files "
        "and dataset path configuration."
    )
    st.exception(error)
    st.stop()

st.caption(f"Data source: {data_source}")

total_images = int(df["Images"].sum())
categories = int(df["Category"].nunique())
normal_images = int(df.loc[df["Class"] == "good", "Images"].sum())
defective_images = int(df.loc[df["Class"] != "good", "Images"].sum())

show_kpi_cards(
    [
        ("Images", f"{total_images:,}"),
        ("Categories", str(categories)),
        ("Normal Images", f"{normal_images:,}"),
        ("Defective Images", f"{defective_images:,}"),
    ]
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Image Counts",
        "Good vs Defective",
        "Defect Diversity",
        "Defect Severity",
        "RGB Analysis",
    ]
)

with tab1:
    st.caption(
        "The number of available images differs between categories, "
        "which introduces category-level imbalance."
    )
    figure = plot_total_images_per_category(df)
    st.pyplot(figure, width="stretch")
    plt.close(figure)

with tab2:
    st.caption(
        "Good images are more frequent than defective images in every category."
    )
    figure = plot_good_vs_defect_distribution(df)
    st.pyplot(figure, width="stretch")
    plt.close(figure)

with tab3:
    st.caption(
        "The number of available defect types varies strongly between categories."
    )
    figure = plot_defect_types_per_category(df)
    st.pyplot(figure, width="stretch")
    plt.close(figure)

with tab4:
    severity_tab, area_tab = st.tabs(["Severity by Category", "Area Distribution"])

    with severity_tab:
        st.caption(
            "Ground-truth masks show that defect severity varies considerably "
            "between categories."
        )
        _show_optional_plot(
            Path(DEFECT_STATISTICS_PATH),
            plot_defect_severity,
            "The precomputed defect statistics file is not available in this deployment.",
        )

    with area_tab:
        st.caption(
            "Defect areas span a broad range, from very small local anomalies "
            "to large damaged regions."
        )
        _show_optional_plot(
            Path(DEFECT_STATISTICS_PATH),
            plot_defect_area_distribution,
            "The precomputed defect statistics file is not available in this deployment.",
        )
        st.info(
            """
            The logarithmic scale reveals both very small and very large defects.
            Each value is calculated from a real ground-truth mask.
            """
        )

with tab5:
    st.caption(
        "Solid curves represent normal images; dashed curves represent anomalous images."
    )
    _show_optional_plot(
        Path(RGB_DENSITY_PATH),
        plot_rgb_intensity_distribution,
        "The precomputed RGB density file is not available in this deployment.",
    )
    st.info(
        """
        Normal and anomalous RGB distributions overlap strongly. Pixel intensity
        alone cannot reliably separate normal and defective samples. The anomaly
        signal depends more strongly on spatial structure, texture and local consistency.
        """
    )

with st.expander("EDA Conclusion", expanded=False):
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
        These findings motivate deep feature-based, patch-level anomaly detection
        methods that preserve spatial information.
        """
    )