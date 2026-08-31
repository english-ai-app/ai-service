from functools import lru_cache
from typing import List

from ultralytics import YOLO

from src.core.config import get_settings
from src.schemas.detection import BoundingBox, DetectedLabel


@lru_cache(maxsize=1)
def get_model() -> YOLO:
    settings = get_settings()
    return YOLO(settings.yolo_model_path)


def detect_labels(image_path: str) -> List[DetectedLabel]:
    settings = get_settings()
    model = get_model()
    results = model.predict(image_path, conf=settings.yolo_confidence, verbose=False)
    labels = []

    for result in results:
        names = result.names
        image_height, image_width = result.orig_shape
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            labels.append(
                DetectedLabel(
                    label=names[class_id],
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


def merge_duplicate_labels(labels: List[DetectedLabel]) -> List[DetectedLabel]:
    best_by_label = {}
    for label in labels:
        current = best_by_label.get(label.label)
        if current is None or label.confidence > current.confidence:
            best_by_label[label.label] = label
    return sorted(best_by_label.values(), key=lambda item: item.confidence, reverse=True)
