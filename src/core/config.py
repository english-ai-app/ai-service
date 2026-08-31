import os
from functools import lru_cache


class Settings:
    def __init__(self):
        self.yolo_model_path = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
        self.yolo_confidence = float(os.getenv("YOLO_CONFIDENCE", "0.25"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
