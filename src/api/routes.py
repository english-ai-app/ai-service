import os

from fastapi import APIRouter

from src.schemas.detection import DetectImageRequest, DetectImageResponse
from src.services.image_loader import get_image_path
from src.services.yolo_detector import detect_labels


router = APIRouter()


@router.get("/health")
def health():
    return {"status": "UP"}


@router.post("/detect-image", response_model=DetectImageResponse)
async def detect_image(request: DetectImageRequest):
    image_path = await get_image_path(request)
    try:
        labels = detect_labels(image_path)
        return DetectImageResponse(labels=labels)
    finally:
        try:
            os.remove(image_path)
        except OSError:
            pass
