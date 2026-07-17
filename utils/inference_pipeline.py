from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import torch
import torchvision.transforms as T
from anomalib.deploy import OpenVINOInferencer
from huggingface_hub import snapshot_download
from PIL import Image


LABEL_MAP = {
    0: "bottle",
    1: "cable",
    2: "capsule",
    3: "carpet",
    4: "grid",
    5: "hazelnut",
    6: "leather",
    7: "metal_nut",
    8: "pill",
    9: "screw",
    10: "tile",
    11: "toothbrush",
    12: "transistor",
    13: "wood",
    14: "zipper",
}

DEFECT_TYPES = {
    "bottle": [
        "broken_large",
        "broken_small",
        "contamination",
    ],
    "cable": [
        "bent_wire",
        "cable_swap",
        "combined",
        "cut_inner_insulation",
        "cut_outer_insulation",
        "missing_cable",
        "missing_wire",
        "poke_insulation",
    ],
    "capsule": [
        "crack",
        "faulty_imprint",
        "poke",
        "scratch",
        "squeeze",
    ],
    "carpet": [
        "color",
        "cut",
        "hole",
        "metal_contamination",
        "thread",
    ],
    "grid": [
        "bent",
        "broken",
        "glue",
        "metal_contamination",
        "thread",
    ],
    "hazelnut": [
        "crack",
        "cut",
        "hole",
        "print",
    ],
    "leather": [
        "color",
        "cut",
        "fold",
        "glue",
        "poke",
    ],
    "metal_nut": [
        "bent",
        "color",
        "flip",
        "scratch",
    ],
    "pill": [
        "color",
        "combined",
        "contamination",
        "crack",
        "faulty_imprint",
        "pill_type",
        "scratch",
    ],
    "screw": [
        "manipulated_front",
        "scratch_head",
        "scratch_neck",
        "thread_side",
        "thread_top",
    ],
    "tile": [
        "crack",
        "glue_strip",
        "gray_stroke",
        "oil",
        "rough",
    ],
    "toothbrush": [
        "defective",
    ],
    "transistor": [
        "bent_lead",
        "cut_lead",
        "damaged_case",
        "misplaced",
    ],
    "wood": [
        "color",
        "combined",
        "hole",
        "liquid",
        "scratch",
    ],
    "zipper": [
        "broken_teeth",
        "combined",
        "fabric_border",
        "fabric_interior",
        "rough",
        "split_teeth",
        "squeezed_teeth",
    ],
}


def load_dinov2_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vitl14",
    ).to(device).eval()

    return model, device


def extract_hybrid_features(
    image: Image.Image,
    model,
    device,
) -> np.ndarray:

    transform = T.Compose(
        [
            T.Resize((518, 518)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    tensor = (
        transform(image.convert("RGB"))
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        layer_output = model.get_intermediate_layers(
            tensor,
            n=1,
            return_class_token=True,
        )

        patch_tokens, cls_token = layer_output[0]

        patch_features = torch.mean(
            patch_tokens,
            dim=1,
        )

        hybrid_feature = torch.cat(
            (
                cls_token,
                patch_features,
            ),
            dim=1,
        )

    return (
        hybrid_feature
        .cpu()
        .numpy()
        .flatten()
    )


def load_patchcore_model(
    category: str,
) -> OpenVINOInferencer:

    repo_id = (
        "CodingBricks/"
        "liora_mvtec_ad_project"
    )

    subfolder = (
        f"patchcore_{category}/"
        "weights/openvino"
    )

    local_dir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=f"{subfolder}/*",
    )

    xml_path = (
        Path(local_dir)
        / subfolder
        / "model.xml"
    )

    return OpenVINOInferencer(
        path=xml_path
    )


def create_heatmap(
    anomaly_map: np.ndarray,
    original_image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert the PatchCore anomaly map into:
    1. a color heatmap;
    2. an overlay on the original RGB image.
    """

    raw_map = np.asarray(
        anomaly_map,
        dtype=np.float32,
    ).squeeze()

    if raw_map.ndim != 2:
        raise ValueError(
            "Expected a two-dimensional anomaly map, "
            f"but received shape {raw_map.shape}."
        )

    minimum = float(raw_map.min())
    maximum = float(raw_map.max())

    if maximum > minimum:
        normalized_map = (
            raw_map - minimum
        ) / (
            maximum - minimum
        )
    else:
        normalized_map = np.zeros_like(
            raw_map,
            dtype=np.float32,
        )

    heatmap_uint8 = (
        normalized_map * 255
    ).clip(
        0,
        255,
    ).astype(
        np.uint8
    )

    heatmap_bgr = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    heatmap_rgb = cv2.cvtColor(
        heatmap_bgr,
        cv2.COLOR_BGR2RGB,
    )

    height, width = original_image.shape[:2]

    heatmap_rgb = cv2.resize(
        heatmap_rgb,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    overlay = cv2.addWeighted(
        original_image,
        0.65,
        heatmap_rgb,
        0.35,
        0,
    )

    return heatmap_rgb, overlay

def run_inspection(
    image: Image.Image,
    app_root: str | Path,
    dino_model,
    device,
) -> dict[str, Any]:

    app_root = Path(app_root)

    feature = extract_hybrid_features(
        image=image,
        model=dino_model,
        device=device,
    )

    router_path = (
        app_root
        / "models"
        / "object_model"
        / "object_type_router.joblib"
    )

    router = joblib.load(
        router_path
    )

    object_prediction = router.predict(
        [feature]
    )

    object_type = LABEL_MAP[
        int(object_prediction[0])
    ]

    patchcore = load_patchcore_model(
        object_type
    )

    image_rgb = np.asarray(
        image.convert("RGB")
    )

    prediction = patchcore.predict(
        image=image_rgb
    )

    label_array = np.asarray(
    prediction.pred_label
    ).reshape(-1)

    if label_array.size == 0:
        raise ValueError(
            "PatchCore returned an empty prediction label."
        )

    is_anomaly = bool(
        label_array[0]
    )

    
    score_array = np.asarray(
    prediction.pred_score,
    dtype=np.float64,
    ).reshape(-1)

    if score_array.size == 0:
        raise ValueError(
            "PatchCore returned an empty anomaly score."
        )

    anomaly_score = float(
        score_array[0]
    )

    heatmap = None
    overlay= None
    defect_type = None

    if (
        is_anomaly
        and prediction.anomaly_map is not None
        ):
        heatmap, overlay = create_heatmap(
        anomaly_map=prediction.anomaly_map,
        original_image=image_rgb,
        )

    if is_anomaly:

        svm_path = (
            app_root
            / "models"
            / "defect_models"
            / (
                f"dinov2_svm_"
                f"{object_type}_final.joblib"
            )
        )

        if svm_path.exists():

            defect_svm = joblib.load(
                svm_path
            )

            defect_prediction = (
                defect_svm.predict(
                    [feature]
                )
            )

            defect_index = int(
                defect_prediction[0]
            )

            defect_type = (
                DEFECT_TYPES[object_type][
                    defect_index
                ]
            )

    return {
        "object_type": object_type,
        "is_anomaly": is_anomaly,
        "anomaly_score": anomaly_score,
        "defect_type": defect_type,
        "heatmap": heatmap,
        "overlay": overlay,
    }