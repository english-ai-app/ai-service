from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class DetectImageRequest(BaseModel):
    imageUrl: Optional[str] = None
    imageBase64: Optional[str] = None
    imageContentType: Optional[str] = Field(default="image/jpeg")
    imageFileName: Optional[str] = None

    @model_validator(mode="after")
    def validate_image_source(self):
        if not self.imageUrl and not self.imageBase64:
            raise ValueError("imageUrl or imageBase64 is required")
        return self


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class DetectedLabel(BaseModel):
    label: str
    confidence: float
    boundingBox: BoundingBox


class DetectImageResponse(BaseModel):
    labels: List[DetectedLabel]
