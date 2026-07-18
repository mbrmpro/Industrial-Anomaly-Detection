#print("EDA module loaded")
import os
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from utils.config import (
    FIGURE_WIDTH,
    FIGURE_HEIGHT,
    FIGURE_DPI,
)

# ==========================================================
# Dataset
# .....
# ==========================================================
def scan_mvtec_dataset(dataset_root):
    """
    Scan the MVTec dataset and return a DataFrame with
    category, dataset split, class and image count.
    """

    dataset_root = Path(dataset_root)

    data = []

    for category in dataset_root.iterdir():

        if not category.is_dir():
            continue

        # -----------------------------
        # TRAIN
        # -----------------------------

        train_path = category / "train"

        if train_path.exists():

            for cls in train_path.iterdir():

                if cls.is_dir():

                    count = len(list(cls.glob("*.png")))

                    data.append({

                        "Category": category.name,

                        "Dataset": "train",

                        "Class": cls.name,

                        "Images": count
                    })

        # -----------------------------
        # TEST
        # -----------------------------

        test_path = category / "test"

        if test_path.exists():

            for cls in test_path.iterdir():

                if cls.is_dir():

                    count = len(list(cls.glob("*.png")))

                    data.append({

                        "Category": category.name,

                        "Dataset": "test",

                        "Class": cls.name,

                        "Images": count
                    })

    return pd.DataFrame(data)

# ==========================================================
# DIAGRAM 1
# .....
# ==========================================================
def plot_train_test_distribution(df):
    """
    Overall Train/Test distribution.
    Returns matplotlib figure.
    """

    dataset_totals = df.groupby("Dataset")["Images"].sum()

    train_count = dataset_totals.get("train", 0)

    test_count = dataset_totals.get("test", 0)

    total = train_count + test_count

    train_pct = train_count / total * 100

    test_pct = test_count / total * 100

    fig, ax = plt.subplots(
    figsize=(
        FIGURE_WIDTH,
        FIGURE_HEIGHT,
    ),
    dpi=FIGURE_DPI,
    )

    ax.barh(
        y=0,
        width=train_count,
        color="#4C72B0",
        height=0.45
    )

    ax.barh(
        y=0,
        width=test_count,
        left=train_count,
        color="#DD8452",
        height=0.45
    )

    ax.text(

        train_count/2,

        0,

        f"Train\n{train_pct:.1f}%",

        ha="center",

        va="center",

        color="white",

        fontsize=11,

        fontweight="bold"
    )

    ax.text(

        train_count+test_count/2,

        0,

        f"Test\n{test_pct:.1f}%",

        ha="center",

        va="center",

        color="white",

        fontsize=11,

        fontweight="bold"
    )

    ax.set_title(

        "Overall Train/Test Distribution",

        fontsize=12,

        fontweight="bold"
    )

    ax.set_xlim(0,total)

    ax.set_xticks([])

    ax.set_yticks([])

    for spine in ax.spines.values():

        spine.set_visible(False)

    plt.tight_layout()

    fig.tight_layout(
    pad=0.5
    )
    return fig

# ==========================================================
# DIAGRAM 2
# .....
# ==========================================================
def plot_category_distribution(df):
    """
    Plot the train/test distribution for each MVTec category.
    Returns a matplotlib Figure.
    """

    df_all_categories = (
        df.groupby(["Category", "Dataset"])["Images"]
        .sum()
        .reset_index()
    )

    category_order = sorted(df_all_categories["Category"].unique())

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(
    figsize=(
        FIGURE_WIDTH,
        FIGURE_HEIGHT,
    ),
    dpi=FIGURE_DPI,
    )

    sns.barplot(
        data=df_all_categories,
        x="Category",
        y="Images",
        hue="Dataset",
        order=category_order,
        palette={
            "train": "#4C72B0",
            "test": "#DD8452"
        },
        edgecolor="black",
        linewidth=0.7,
        ax=ax
    )

    # ---------------------------------------------------
    # Add labels above bars
    # ---------------------------------------------------

    for p in ax.patches:

        height = p.get_height()

        if height > 0:

            ax.annotate(

                f"{int(height)}",

                (
                    p.get_x() + p.get_width()/2,
                    height
                ),

                ha="center",

                va="bottom",

                fontsize=9,

                fontweight="bold",

                xytext=(0,4),

                textcoords="offset points"
            )

    ax.set_title(
        "Train/Test Distribution per MVTec Category",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Category",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Number of Images",
        fontsize=12,
        fontweight="bold"
    )

    plt.xticks(
        rotation=35,
        ha="right"
    )

    plt.tight_layout()
    fig.tight_layout(
    pad=0.5
    )
    return fig

# ==========================================================
# DIAGRAM 3
# Total Images per Category
# ==========================================================

def plot_total_images_per_category(df):

    import matplotlib.pyplot as plt
    import seaborn as sns

    TEXTURES = [
        "carpet",
        "grid",
        "leather",
        "tile",
        "wood"
    ]

    df_totals = (
        df.groupby("Category")["Images"]
        .sum()
        .reset_index()
    )

    df_totals["Type"] = df_totals["Category"].apply(
        lambda x: "Texture" if x in TEXTURES else "Product"
    )

    df_totals = df_totals.sort_values(
        by="Images",
        ascending=False
    )

    sns.set_style("whitegrid")

    fig, ax = plt.subplots(
    figsize=(
        FIGURE_WIDTH,
        FIGURE_HEIGHT,
    ),
    dpi=FIGURE_DPI,
    )

    sns.barplot(
        data=df_totals,
        x="Category",
        y="Images",
        hue="Type",
        dodge=False,
        palette={
            "Product":"#4C72B0",
            "Texture":"#55A868"
        },
        edgecolor="black",
        linewidth=0.7,
        ax=ax
    )

    for p in ax.patches:

        height = p.get_height()

        if height > 0:

            ax.annotate(

                f"{int(height)}",

                (
                    p.get_x()+p.get_width()/2,
                    height
                ),

                ha="center",

                xytext=(0,6),

                textcoords="offset points",

                fontsize=9,

                fontweight="bold"

            )

    ax.set_title(
        "Number of Images per Category",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_xlabel("Category")
    ax.set_ylabel("Images")

    plt.xticks(
        rotation=35,
        ha="right"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.tight_layout(
    pad=0.5
)
    return fig

# ==========================================================
# DIAGRAM 4
# Good vs Defect Images per Category
# ==========================================================

def plot_good_vs_defect_distribution(df):

    import matplotlib.pyplot as plt

    df_plot = df.copy()

    df_plot["Label_Type"] = df_plot["Class"].apply(
        lambda x: "Good" if x == "good" else "Defect"
    )

    stacked = (
        df_plot
        .groupby(["Category", "Label_Type"])["Images"]
        .sum()
        .unstack(fill_value=0)
    )

    stacked["Total"] = stacked.sum(axis=1)

    stacked = stacked.sort_values(
        "Total",
        ascending=False
    )

    fig, ax = plt.subplots(
    figsize=(
        FIGURE_WIDTH,
        FIGURE_HEIGHT,
    ),
    dpi=FIGURE_DPI,
    )

    stacked[["Defect","Good"]].plot(
        kind="bar",
        stacked=True,
        color=["#4C72B0","#FFB347"],
        edgecolor="black",
        linewidth=0.5,
        width=0.8,
        ax=ax
    )

    for i,total in enumerate(stacked["Total"]):

        ax.text(
            i,
            total+5,
            str(int(total)),
            ha="center",
            fontsize=9,
            fontweight="bold"
        )

    ax.set_title(
        "Distribution of Good and Defective Images",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_xlabel("")
    ax.set_ylabel("Images")

    plt.xticks(
        rotation=35,
        ha="right"
    )

    ax.legend(
        title="Image Type",
        frameon=False
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=.30
    )

    plt.tight_layout()
    fig.tight_layout(
    pad=0.5
    )

    return fig

# ==========================================================
# DIAGRAM 5
# Number of Defect Types per Category
# ==========================================================

def plot_defect_types_per_category(df):

    import matplotlib.pyplot as plt

    df_defects = df[
        (df["Dataset"] == "test") &
        (df["Class"] != "good")
    ].copy()

    defect_counts = (
        df_defects
        .groupby("Category")["Class"]
        .nunique()
        .reset_index()
    )

    defect_counts.columns = [
        "Category",
        "Defect Types"
    ]

    defect_counts = defect_counts.sort_values(
        by="Defect Types",
        ascending=False
    )

    fig, ax = plt.subplots(
    figsize=(
        FIGURE_WIDTH,
        FIGURE_HEIGHT,
    ),
    dpi=FIGURE_DPI,
    )

    bars = ax.bar(
        defect_counts["Category"],
        defect_counts["Defect Types"],
        color="#C44E52",
        edgecolor="black",
        linewidth=0.6
    )

    for bar in bars:

        h = bar.get_height()

        ax.text(
            bar.get_x()+bar.get_width()/2,
            h+0.08,
            str(int(h)),
            ha="center",
            fontsize=9,
            fontweight="bold"
        )

    ax.set_title(
        "Number of Defect Types per Category",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_xlabel("")
    ax.set_ylabel("Defect Types")

    ax.set_ylim(
        0,
        defect_counts["Defect Types"].max()+1
    )

    plt.xticks(
        rotation=35,
        ha="right"
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=.30
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.tight_layout(
    pad=0.5
    )
    return fig

# ==========================================================
# DIAGRAM 6
# Defect Severity (Ground Truth Masks)
# ==========================================================
def plot_defect_severity(defect_statistics_path):
    """
    Plot mean defect-area percentage per category.

    Bubble size represents the standard deviation of
    defect-area percentages.

    If the required mask statistic is unavailable, return
    an informative placeholder figure instead of crashing.
    """

    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    statistics_path = Path(defect_statistics_path)

    if not statistics_path.exists():
        raise FileNotFoundError(
            f"Defect statistics not found: {statistics_path}"
        )

    df = pd.read_csv(statistics_path)

    required_columns = {
        "category",
        "defect_area_percent",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        fig, ax = plt.subplots(
            figsize=(
                FIGURE_WIDTH,
                FIGURE_HEIGHT,
            ),
            dpi=FIGURE_DPI,
        )

        ax.text(
            0.5,
            0.58,
            "Defect Severity unavailable",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            transform=ax.transAxes,
        )

        ax.text(
            0.5,
            0.43,
            (
                "The deployment statistics file does not contain\n"
                "ground-truth mask area percentages."
            ),
            ha="center",
            va="center",
            fontsize=10,
            transform=ax.transAxes,
        )

        ax.text(
            0.5,
            0.28,
            (
                "Required column: defect_area_percent\n"
                f"Available columns: {', '.join(df.columns)}"
            ),
            ha="center",
            va="center",
            fontsize=9,
            transform=ax.transAxes,
        )

        ax.set_axis_off()
        fig.tight_layout(pad=0.5)

        return fig

    summary = (
        df.groupby("category")["defect_area_percent"]
        .agg(["mean", "std"])
        .fillna(0)
        .sort_values("mean")
    )

    categories = summary.index.tolist()
    mean_values = summary["mean"].to_numpy()
    std_values = summary["std"].to_numpy()

    bubble_sizes = np.maximum(
        std_values * 350,
        60,
    )

    fig, ax = plt.subplots(
        figsize=(
            FIGURE_WIDTH,
            FIGURE_HEIGHT,
        ),
        dpi=FIGURE_DPI,
    )

    scatter = ax.scatter(
        mean_values,
        categories,
        s=bubble_sizes,
        c=mean_values,
        cmap="inferno",
        alpha=0.78,
        edgecolors="black",
        linewidth=0.8,
    )

    overall_mean = float(
        df["defect_area_percent"].mean()
    )

    ax.axvline(
        overall_mean,
        color="#D32F2F",
        linestyle="--",
        linewidth=2,
        label=f"Overall Mean ({overall_mean:.2f}%)",
    )

    for index, value in enumerate(mean_values):
        ax.text(
            value + 0.20,
            index,
            f"{value:.2f}%",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    colorbar = fig.colorbar(
        scatter,
        ax=ax,
        pad=0.02,
    )

    colorbar.set_label(
        "Mean Defect Area (%)",
        fontweight="bold",
    )

    ax.set_title(
        "Defect Severity per Category",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel(
        "Mean Defect Area (%)",
        fontweight="bold",
    )

    ax.set_ylabel(
        "Category",
        fontweight="bold",
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.3,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False)

    fig.tight_layout(pad=0.5)
# ==========================================================
    return fig
# DIAGRAM 7
# Real Defect Area Distribution on Logarithmic Scale
# ==========================================================

def plot_defect_area_distribution(
    defect_statistics_path
):
    """
    Plot the distribution of real defect areas using
    precomputed ground-truth-mask statistics.
    """

    from pathlib import Path

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    import pandas as pd

    statistics_path = Path(
        defect_statistics_path
    )

    if not statistics_path.exists():

        raise FileNotFoundError(
            f"Defect statistics not found: "
            f"{statistics_path}"
        )

    df = pd.read_csv(
        statistics_path
    )

    defect_areas = (
        df["defect_area_pixels"]
        .dropna()
        .to_numpy(dtype=np.float64)
    )

    defect_areas = defect_areas[
        defect_areas > 0
    ]

    if len(defect_areas) == 0:

        raise ValueError(
            "No positive defect-area values found."
        )

    minimum = max(
        1.0,
        float(defect_areas.min())
    )

    maximum = float(
        defect_areas.max()
    )

    bins = np.logspace(
        np.log10(minimum),
        np.log10(maximum),
        40
    )

    median_area = float(
        np.median(defect_areas)
    )

    fig, ax = plt.subplots(
    figsize=(
        FIGURE_WIDTH,
        FIGURE_HEIGHT,
    ),
    dpi=FIGURE_DPI,
    )

    ax.hist(
        defect_areas,
        bins=bins,
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
        label=(
            f"Ground-truth masks "
            f"({len(defect_areas):,})"
        )
    )

    ax.set_xscale("log")

    ax.axvline(
        median_area,
        color="#D32F2F",
        linestyle="--",
        linewidth=2,
        label=(
            f"Median "
            f"({median_area:,.0f} px²)"
        )
    )

    ax.xaxis.set_major_formatter(
        ticker.LogFormatterMathtext()
    )

    ax.set_title(
        "Distribution of Defect Areas "
        "on a Logarithmic Scale",
        fontsize=12,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel(
        "Defect Area (pixels²)",
        fontweight="bold"
    )

    ax.set_ylabel(
        "Number of Ground-Truth Masks",
        fontweight="bold"
    )

    ax.grid(
        axis="x",
        which="both",
        linestyle="--",
        alpha=0.20
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.30
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False
    )

    
    fig.tight_layout(
    pad=0.5
    )
    return fig
# ==========================================================
# DIAGRAM 8
# Cached Real RGB Intensity Distribution
# ==========================================================
# ==========================================================
# DIAGRAM 8
# Cached Real RGB Intensity Distribution
# ==========================================================
def plot_rgb_intensity_distribution(rgb_density_path):
    """
    Load precomputed RGB histograms and display normal versus
    anomalous RGB intensity distributions.

    Streamlit performs no image loading and no KDE calculation.
    """

    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    rgb_density_path = Path(rgb_density_path)

    if not rgb_density_path.exists():
        raise FileNotFoundError(
            f"RGB density file not found: {rgb_density_path}"
        )

    with np.load(rgb_density_path) as data:

        intensities = data["intensities"]

        smooth_curves = {
            "R normal": data["r_normal"],
            "R anomalous": data["r_anomalous"],
            "G normal": data["g_normal"],
            "G anomalous": data["g_anomalous"],
            "B normal": data["b_normal"],
            "B anomalous": data["b_anomalous"],
        }

        raw_curves = {
            "R normal": data["r_normal_raw"],
            "R anomalous": data["r_anomalous_raw"],
            "G normal": data["g_normal_raw"],
            "G anomalous": data["g_anomalous_raw"],
            "B normal": data["b_normal_raw"],
            "B anomalous": data["b_anomalous_raw"],
        }

        normal_image_count = int(
            data["normal_image_count"]
        )

        anomalous_image_count = int(
            data["anomalous_image_count"]
        )

    channel_colors = {
        "R": "#E74C3C",
        "G": "#2ECC71",
        "B": "#2980B9",
    }

    # Important:
    # Use plt.figure(), not plt.subplots(),
    # because the axes are positioned manually.
    fig = plt.figure(
        figsize=(
            FIGURE_WIDTH,
            FIGURE_HEIGHT,
        ),
        dpi=FIGURE_DPI,
    )

    ax = fig.add_axes(
        [0.07, 0.16, 0.67, 0.70]
    )

    ax_zoom = fig.add_axes(
        [0.79, 0.23, 0.18, 0.50]
    )

    # Main smoothed RGB curves
    for label, density in smooth_curves.items():

        channel = label[0]
        is_anomalous = "anomalous" in label

        linestyle = "--" if is_anomalous else "-"
        color = channel_colors[channel]

        ax.plot(
            intensities,
            density,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=label,
        )

    # Raw values in the zoom preserve the exact 255 peak
    for label, density in raw_curves.items():

        channel = label[0]
        is_anomalous = "anomalous" in label

        linestyle = "--" if is_anomalous else "-"
        color = channel_colors[channel]

        ax_zoom.plot(
            intensities,
            density,
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
        )

    # ------------------------------------------------------
    # Main axis
    # ------------------------------------------------------

    ax.set_title(
        "RGB Intensity Distribution — All Categories\n"
        "Solid = Normal · Dashed = Anomalous",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )

    ax.set_xlabel(
        "Pixel Intensity (0–255)",
        fontsize=9,
        fontweight="bold",
    )

    ax.set_ylabel(
        "Relative Pixel Frequency",
        fontsize=9,
        fontweight="bold",
    )

    ax.set_xlim(
        0,
        255,
    )

    ax.tick_params(
        axis="both",
        labelsize=8,
    )

    ax.axvspan(
        250,
        255,
        color="#EF4444",
        alpha=0.06,
    )

    ax.grid(
        linestyle="--",
        alpha=0.25,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        title="Channel / Image Type",
        ncol=2,
        frameon=False,
        loc="upper center",
        fontsize=7.5,
        title_fontsize=8,
    )

    ax.text(
        0.99,
        0.96,
        (
            f"Normal images: {normal_image_count:,}\n"
            f"Anomalous images: {anomalous_image_count:,}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#CBD5E1",
            "alpha": 0.9,
        },
    )

    # ------------------------------------------------------
    # Zoom axis
    # ------------------------------------------------------

    zoom_mask = intensities >= 230

    zoom_maximum = max(
        float(curve[zoom_mask].max())
        for curve in raw_curves.values()
    )

    ax_zoom.set_xlim(
        230,
        255,
    )

    ax_zoom.set_ylim(
        0,
        zoom_maximum * 1.12,
    )

    ax_zoom.set_title(
        "High Intensities\n(230–255)",
        fontsize=8,
        fontweight="bold",
        pad=5,
    )

    ax_zoom.set_xlabel(
        "Intensity",
        fontsize=7.5,
    )

    ax_zoom.set_ylabel(
        "Relative Frequency",
        fontsize=7.5,
    )

    ax_zoom.tick_params(
        axis="both",
        labelsize=7,
    )

    ax_zoom.grid(
        linestyle="--",
        alpha=0.25,
    )

    ax_zoom.axvline(
        255,
        color="#D32F2F",
        linestyle=":",
        linewidth=1.2,
    )

    ax_zoom.text(
        0.05,
        0.92,
        "Peak near 255:\nbright / white backgrounds",
        transform=ax_zoom.transAxes,
        va="top",
        fontsize=6.8,
        color="#B91C1C",
        fontweight="bold",
    )

    return fig

#################################################
# load dataset
#################################################
def load_dataset_summary(
    summary_path,
    local_dataset_path=None
):
    """
    Load the precomputed dataset summary.

    If the CSV is unavailable locally, the dataset can optionally
    be scanned directly as a fallback.
    """

    from pathlib import Path
    import pandas as pd

    summary_path = Path(summary_path)

    if summary_path.exists():
        return pd.read_csv(summary_path)

    if local_dataset_path is not None:

        local_dataset_path = Path(
            local_dataset_path
        )

        if local_dataset_path.exists():
            return scan_mvtec_dataset(
                local_dataset_path
            )

    raise FileNotFoundError(
        "Neither the dataset summary nor the local "
        "dataset could be found."
    )