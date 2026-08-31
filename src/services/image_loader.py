import base64
import binascii
import re
import tempfile

import httpx
from fastapi import HTTPException

from src.schemas.detection import DetectImageRequest


async def get_image_path(request: DetectImageRequest) -> str:
    if request.imageBase64:
        return write_base64_image(
            request.imageBase64,
            request.imageContentType or "image/jpeg",
        )

    return await download_image(request.imageUrl or "")


def write_base64_image(image_base64: str, content_type: str) -> str:
    try:
        normalized_base64 = re.sub(r"^data:image/[^;]+;base64,", "", image_base64.strip())
        image_bytes = base64.b64decode(normalized_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}") from exc

    suffix = guess_image_suffix(content_type)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(image_bytes)
        return temp_file.name


async def download_image(image_url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(image_url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=400, detail=f"Cannot download image: {exc}") from exc

    content_type = response.headers.get("content-type", "")
    if "image" not in content_type:
        raise HTTPException(status_code=400, detail="URL does not point to an image")

    suffix = guess_image_suffix(content_type)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(response.content)
        return temp_file.name


def guess_image_suffix(content_type: str) -> str:
    if "png" in content_type:
        return ".png"
    if "webp" in content_type:
        return ".webp"
    return ".jpg"
