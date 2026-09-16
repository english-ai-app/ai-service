import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def train_lion_tiger_detector() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLOv10 lion/tiger detector.")
    parser.add_argument(
        "--data",
        default="Dataset/lion.v1-version-1.yolov8/data.yaml",
        help="Path to Roboflow/YOLO data.yaml.",
    )
    parser.add_argument("--model", default="weights/base/yolov10n.pt", help="Base model to fine-tune.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="Training device, e.g. 0 for first GPU or cpu.")
    parser.add_argument("--project", default="training/runs/lion_tiger")
    parser.add_argument("--name", default="yolov10n_lion_tiger")
    parser.add_argument("--output", default="weights/lion_tiger_best.pt")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    model = YOLO(args.model)
    result = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )

    best_model = Path(result.save_dir) / "weights" / "best.pt"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_model, output_path)

    print(f"Best model copied to: {output_path}")
    print("Set SPECIAL_MODEL_PATH to this path, or keep the default if unchanged.")


if __name__ == "__main__":
    train_lion_tiger_detector()
