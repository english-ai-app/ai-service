import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings:
    def __init__(self):
        self.yolo_model_path = os.getenv("YOLO_MODEL_PATH", "weights/base/yolov10n.pt")
        self.yolo_confidence = float(os.getenv("YOLO_CONFIDENCE", "0.35"))
        self.yolo_image_size = int(os.getenv("YOLO_IMAGE_SIZE", "960"))
        self.yolo_max_detections = int(os.getenv("YOLO_MAX_DETECTIONS", "20"))
        self.special_detector_enabled = parse_bool(os.getenv("SPECIAL_DETECTOR_ENABLED", "true"))
        self.special_model_path = os.getenv("SPECIAL_MODEL_PATH", "weights/lion_tiger_best.pt")
        self.special_model_paths = parse_csv(os.getenv("SPECIAL_MODEL_PATHS", self.special_model_path))
        self.special_confidence = float(os.getenv("SPECIAL_CONFIDENCE", "0.70"))
        self.special_labels = {
            item.strip().lower()
            for item in os.getenv("SPECIAL_LABELS", "lion,tiger").split(",")
            if item.strip()
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
