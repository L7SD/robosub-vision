# RoboSub Vision

Tools for checking underwater object-detection datasets, splitting capture sessions, inspecting predictions, and running YOLOv7 experiments.

The toolkit runs without a GPU. Detector training and inference use a separate YOLOv7 installation. Underwater images, trained weights, and the final class-name mapping are not included.

## Run the example

Install Python 3.12 and open PowerShell in this folder:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m robosubvision demo --output data/demo
.venv\Scripts\python.exe -m robosubvision validate --data data/demo/dataset.yaml --groups data/demo/groups.csv --output runs/demo-validation.json
.venv\Scripts\python.exe -m robosubvision evaluate --data data/demo/dataset.yaml --predictions data/demo/fixture_predictions --output runs/demo-diagnostics.json
.venv\Scripts\python.exe -m robosubvision overlay --data data/demo/dataset.yaml --image data/demo/val/images/val_001.png --predictions data/demo/fixture_predictions/val_001.txt --output runs/demo-overlay.png
```

On macOS/Linux, create the environment with `python3.12 -m venv .venv` and use `.venv/bin/python` for the remaining commands.

The example contains synthetic shapes and prediction fixtures with one missed object and one duplicate detection. Expected totals are 3 true positives, 1 false positive, and 1 false negative. These are software checks, not detector benchmarks.

## Dataset layout

Keep corresponding `images` and `labels` directories, such as `train/images/frame001.jpg` and `train/labels/frame001.txt`. Each label line contains:

```text
class_id center_x center_y width height
```

Coordinates are normalized to `[0, 1]`. Empty label files represent background images. This toolkit supports bounding boxes, not segmentation polygons.

Copy `configs/robosub.example.yaml` to `configs/robosub.yaml`. Set its dataset path and replace the eight placeholder names with the correct names in label-ID order.

Capture groups can be recorded in a CSV:

```csv
image,group_id
train/images/frame001.jpg,pool_session_01
train/images/frame002.jpg,pool_session_01
val/images/frame120.jpg,pool_session_02
```

Paths are relative to the dataset root configured in YAML. Frames from the same recording or capture session should share a group.

```powershell
.venv\Scripts\python.exe -m robosubvision validate --data configs/robosub.yaml --groups data/robosub/groups.csv --output runs/dataset-check.json
```

Validation checks label fields, class ranges, box bounds, image readability, duplicate images across splits, and capture-group overlap. Missing group metadata produces a warning because exact-image checks cannot detect adjacent-frame leakage.

## Split by capture session

Create `configs/classes.txt` with one class name per line in label-ID order, then run:

```powershell
.venv\Scripts\python.exe -m robosubvision split --root data/robosub --groups data/robosub/groups.csv --names configs/classes.txt --output data/grouped-split --seed 42
```

The command creates approximately 60/20/20 train/validation/test groups and writes image manifests without moving the originals. It does not stratify classes; inspect the class counts before training. Existing output directories must be empty.

## Inspect detections

Prediction files use six fields: `class cx cy width height confidence`. The `evaluate` command matches predictions to ground truth by class and IoU, with at most one prediction per target. Duplicates count as false positives.

Default thresholds are confidence 0.25 and IoU 0.5. Reports contain per-class and micro precision, recall, and F1. These fixed-threshold metrics are different from mAP; use YOLOv7's evaluation script for mAP. Missing prediction files count as zero detections and are reported explicitly.

## Train or run YOLOv7

See [detector setup](docs/YOLOV7_SETUP.md) for the pinned checkout, separate Python environment, and training/inference commands.

Each run records the upstream commit, checkpoint hash, dataset fingerprint, environment, arguments, and console output. The historical experiment is described in the [model card](docs/MODEL_CARD.md), with its notebook retained in `notebooks/robosub_training.ipynb`.

## Code layout

| Module | Responsibility |
|---|---|
| `labels.py` | Label parsing, box validation, and IoU |
| `dataset.py` | Dataset loading, validation, groups, and splits |
| `evaluation.py` | Matching, metrics, and overlays |
| `demo.py` | Synthetic test fixtures |
| `runner.py` | YOLOv7 execution and run records |
| `__main__.py` | Command-line interface |

See [test status](docs/TESTING.md). This repository contains no vehicle-control or ROS node implementation.
