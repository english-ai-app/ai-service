from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

from ultralytics import YOLO

from src.core.config import get_settings
from src.schemas.detection import BoundingBox, DetectedLabel

LABEL_ALIASES = {
    "computer mouse": "mouse",
    "remote control": "remote",
    "mobile phone": "cell phone",
    "phone": "cell phone",
}


@lru_cache(maxsize=1)
def get_base_model() -> YOLO:
    settings = get_settings()
    return YOLO(settings.yolo_model_path)


@lru_cache(maxsize=16)
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def get_special_models() -> List[Tuple[str, YOLO]]:
    settings = get_settings()
    if not settings.special_detector_enabled:
        return []

    models = []
    for model_path_value in settings.special_model_paths:
        model_path = Path(model_path_value)
        if model_path.exists():
            models.append((str(model_path), load_model(str(model_path))))

    return models


def predict_labels(model: YOLO, image_path: str, confidence: float) -> List[DetectedLabel]:
    settings = get_settings()
    results = model.predict(
        image_path,
        conf=confidence,
        imgsz=settings.yolo_image_size,
        max_det=settings.yolo_max_detections,
        verbose=False,
    )
    labels = []

    for result in results:
        names = result.names
        image_height, image_width = result.orig_shape
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = LABEL_ALIASES.get(names[class_id], names[class_id])
            labels.append(
                DetectedLabel(
                    label=label,
                    confidence=round(float(box.conf[0].item()), 4),
                    boundingBox=BoundingBox(
                        x=round(max(x1 / image_width, 0.0), 4),
                        y=round(max(y1 / image_height, 0.0), 4),
                        width=round(max((x2 - x1) / image_width, 0.0), 4),
                        height=round(max((y2 - y1) / image_height, 0.0), 4),
                    ),
                )
            )

    return merge_duplicate_labels(labels)


def detect_special_labels(image_path: str) -> Tuple[List[DetectedLabel], str]:
    settings = get_settings()
    labels: List[DetectedLabel] = []
    sources: List[str] = []

    for model_path, model in get_special_models():
        model_labels = predict_labels(model, image_path, settings.special_confidence)
        special_labels = [label for label in model_labels if label.label.lower() in settings.special_labels]
        if not special_labels:
            continue

        labels.extend(special_labels)
        sources.append(f"special:{model_path}")

    return merge_overlapping_labels(labels), ",".join(sources)


def detect_labels_with_source(image_path: str) -> Tuple[List[DetectedLabel], str]:
    settings = get_settings()
    special_labels, special_source = detect_special_labels(image_path)
    base_labels = predict_labels(get_base_model(), image_path, settings.yolo_confidence)
    labels = merge_overlapping_labels([*special_labels, *base_labels])

    sources = [source for source in [special_source, "base"] if source]
    return labels, "+".join(sources)


def detect_labels(image_path: str) -> List[DetectedLabel]:
    labels, _ = detect_labels_with_source(image_path)
    return labels


def merge_duplicate_labels(labels: List[DetectedLabel]) -> List[DetectedLabel]:
    return merge_overlapping_labels(labels)


def box_iou(first: BoundingBox, second: BoundingBox) -> float:
    first_x2 = first.x + first.width
    first_y2 = first.y + first.height
    second_x2 = second.x + second.width
    second_y2 = second.y + second.height

    intersection_width = max(0.0, min(first_x2, second_x2) - max(first.x, second.x))
    intersection_height = max(0.0, min(first_y2, second_y2) - max(first.y, second.y))
    intersection_area = intersection_width * intersection_height

    first_area = first.width * first.height
    second_area = second.width * second.height
    union_area = first_area + second_area - intersection_area
    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def merge_overlapping_labels(labels: List[DetectedLabel]) -> List[DetectedLabel]:
    settings = get_settings()
    merged: List[DetectedLabel] = []

    for label in sorted(labels, key=lambda item: item.confidence, reverse=True):
        is_duplicate = any(
            box_iou(label.boundingBox, current.boundingBox) >= settings.detection_merge_iou
            for current in merged
        )
        if not is_duplicate:
            merged.append(label)

    return merged
