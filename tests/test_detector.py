import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_IMAGE_SUFFIXES = {
    ".bmp",
    ".dng",
    ".heic",
    ".jpeg",
    ".jpg",
    ".mpo",
    ".pfm",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def iter_image_paths(path: Path) -> List[Path]:
    if path.is_file():
        return [path]

    if path.is_dir():
        return sorted(
            item
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )

    raise FileNotFoundError(f"Image path not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test detector fallback flow on an image or folder.")
    parser.add_argument("path", nargs="?", default="test_images")
    parser.add_argument("--base-model", help="Override YOLO_MODEL_PATH for this test run.")
    parser.add_argument("--special-model", help="Override SPECIAL_MODEL_PATHS for this test run.")
    parser.add_argument("--confidence", help="Override YOLO_CONFIDENCE for this test run.")
    parser.add_argument("--special-confidence", help="Override SPECIAL_CONFIDENCE for this test run.")
    parser.add_argument("--disable-special", action="store_true", help="Run only the base detector.")
    args = parser.parse_args()

    if args.base_model:
        os.environ["YOLO_MODEL_PATH"] = args.base_model
    if args.special_model:
        os.environ["SPECIAL_MODEL_PATHS"] = args.special_model
    if args.confidence:
        os.environ["YOLO_CONFIDENCE"] = args.confidence
    if args.special_confidence:
        os.environ["SPECIAL_CONFIDENCE"] = args.special_confidence
    if args.disable_special:
        os.environ["SPECIAL_DETECTOR_ENABLED"] = "false"

    from src.services.yolo_detector import detect_labels_with_source

    for image_path in iter_image_paths(Path(args.path)):
        labels, source = detect_labels_with_source(str(image_path))
        print(f"\nIMAGE: {image_path}")
        print(f"SOURCE: {source}")
        print(json.dumps([label.model_dump() for label in labels], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
