from pathlib import Path

from utils.config import APP_ROOT, DATASET_PATH


IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
}

CLOUD_SAMPLE_ROOT = (
    APP_ROOT
    / "assets"
    / "dataset_samples"
)


def _image_files(
    directory: Path,
) -> list[Path]:
    """Return supported image files from one directory."""

    if not directory.exists():
        return []

    return sorted(
        path
        for path in directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
        )
    )


def get_categories() -> list[str]:
    """
    Return local dataset categories when available.

    On Streamlit Cloud, fall back to the prepared sample folders.
    """

    local_root = Path(DATASET_PATH)

    if local_root.exists():

        local_categories = sorted(
            folder.name
            for folder in local_root.iterdir()
            if folder.is_dir()
        )

        if local_categories:
            return local_categories

    if CLOUD_SAMPLE_ROOT.exists():

        cloud_categories = sorted(
            folder.name
            for folder in CLOUD_SAMPLE_ROOT.iterdir()
            if folder.is_dir()
        )

        if cloud_categories:
            return cloud_categories

    return []


def get_train_images(
    category: str,
) -> list[Path]:
    """Return local normal training images when available."""

    train_path = (
        Path(DATASET_PATH)
        / category
        / "train"
        / "good"
    )

    return _image_files(
        train_path
    )


def get_test_defects(
    category: str,
) -> list[str]:
    """
    Return local test classes or cloud sample classes.
    """

    local_test_root = (
        Path(DATASET_PATH)
        / category
        / "test"
    )

    if local_test_root.exists():

        local_defects = sorted(
            folder.name
            for folder in local_test_root.iterdir()
            if folder.is_dir()
        )

        if local_defects:
            return local_defects

    cloud_category_root = (
        CLOUD_SAMPLE_ROOT
        / category
    )

    if cloud_category_root.exists():

        cloud_defects = sorted(
            folder.name
            for folder in cloud_category_root.iterdir()
            if folder.is_dir()
        )

        if cloud_defects:
            return cloud_defects

    return []


def get_test_images(
    category: str,
    defect_type: str,
) -> list[Path]:
    """
    Return local test images when available.

    On Streamlit Cloud, return prepared sample images.
    """

    local_test_path = (
        Path(DATASET_PATH)
        / category
        / "test"
        / defect_type
    )

    local_images = _image_files(
        local_test_path
    )

    if local_images:
        return local_images

    cloud_sample_path = (
        CLOUD_SAMPLE_ROOT
        / category
        / defect_type
    )

    return _image_files(
        cloud_sample_path
    )


def count_images(
    category: str,
) -> dict[str, int]:
    """
    Count local dataset images.

    In cloud mode, use available sample-image counts.
    """

    category_root = (
        Path(DATASET_PATH)
        / category
    )

    if category_root.exists():

        train_count = len(
            list(
                (
                    category_root
                    / "train"
                ).rglob("*.png")
            )
        )

        test_count = len(
            list(
                (
                    category_root
                    / "test"
                ).rglob("*.png")
            )
        )

        return {
            "train": train_count,
            "test": test_count,
            "total": (
                train_count
                + test_count
            ),
        }

    cloud_category_root = (
        CLOUD_SAMPLE_ROOT
        / category
    )

    cloud_test_count = 0

    if cloud_category_root.exists():

        cloud_test_count = sum(
            len(
                _image_files(
                    defect_folder
                )
            )
            for defect_folder
            in cloud_category_root.iterdir()
            if defect_folder.is_dir()
        )

    return {
        "train": 0,
        "test": cloud_test_count,
        "total": cloud_test_count,
    }