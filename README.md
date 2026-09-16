# AI Detection Service

## Base YOLOv10 detector

Run the current detector on local test images:

```powershell
.\venv\Scripts\python.exe tests\test_detector.py test_images --disable-special
```

The default base model path is `weights/base/yolov10n.pt`. It detects the built-in COCO classes, including `zebra` and `giraffe`, but not specialized species such as `lion` and `tiger`.

## Lion/Tiger special detector

This project supports an optional special detector for the thesis/demo case:

1. Try one or more fine-tuned special models first.
2. Keep the best special result only if it is at least `SPECIAL_CONFIDENCE`, default `0.70`.
3. If no special model passes that threshold, fall back to the base YOLOv10 detector, default confidence `0.35`.

Default settings:

```env
SPECIAL_DETECTOR_ENABLED=true
SPECIAL_MODEL_PATHS=weights/lion_tiger_best.pt
SPECIAL_CONFIDENCE=0.70
SPECIAL_LABELS=lion,tiger
YOLO_MODEL_PATH=weights/base/yolov10n.pt
YOLO_CONFIDENCE=0.35
```

For multiple trained `.pt` files, separate them with commas:

```env
SPECIAL_MODEL_PATHS=weights/special/lion_tiger_best.pt,weights/special/another_best.pt
```

The API uses `src/services/yolo_detector.py` only for inference. Training and manual model checks live outside `src` so they do not need to be deployed with the backend.

## Train with Roboflow

Use the Roboflow Universe dataset/model:

```text
https://universe.roboflow.com/project-1-7efmg/lion-or-tiger
```

It is an object detection project with 2 classes: `tiger` and `lion`.

Download/export the dataset as YOLO format, then point the training script to its `data.yaml`.

Roboflow CLI example:

```powershell
.\venv\Scripts\python.exe -m pip install roboflow
roboflow login
roboflow download -f yolov8 -l datasets\lion-or-tiger project-1-7efmg/lion-or-tiger/1
```

If the CLI reports `API Key is of Incorrect Type`, login first and paste your Roboflow API key. You can find it in Roboflow account settings.

Example:

```powershell
.\venv\Scripts\python.exe training\train_lion_tiger_detector.py --data Dataset\lion.v1-version-1.yolov8\data.yaml --epochs 50 --imgsz 640 --batch 8
```

The script copies the best checkpoint to:

```text
weights/lion_tiger_best.pt
```

After training, test the full fallback flow:

```powershell
.\venv\Scripts\python.exe tests\test_detector.py test_images --special-model weights\lion_tiger_best.pt
```

To test multiple special models:

```powershell
.\venv\Scripts\python.exe tests\test_detector.py test_images --special-model weights\special\lion_tiger_best.pt --special-model weights\special\another_best.pt
```

You can also test a single image:

```powershell
.\venv\Scripts\python.exe tests\test_detector.py test_images\example.jpg --special-model weights\lion_tiger_best.pt
```

## Deployment layout

Deploy the API code and only the inference weights required at runtime:

```text
src/
weights/base/yolov10n.pt
weights/lion_tiger_best.pt
weights/special/*.pt
requirements.txt
```

Do not deploy training-only data:

```text
Dataset/
runs/
training/
test_images/
tests/
```

For production, this special model should be trained with more images, more backgrounds, different lighting, occlusion, camera angles, and negative examples. The two-class model is useful for demonstrating a special-case improvement in the project.
