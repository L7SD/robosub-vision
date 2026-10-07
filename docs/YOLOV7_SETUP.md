# YOLOv7 setup

The toolkit and detector use separate environments. The pinned YOLOv7 version depends on older NumPy and PyTorch releases.

## Install the detector

Install Git and Python 3.10. From the project root, run:

```powershell
git clone https://github.com/WongKinYiu/yolov7.git external/yolov7
git -C external/yolov7 checkout a207844b1ce82d204ab36d87d496728d3d2348e7
py -3.10 -m venv .venv-yolo
.venv-yolo\Scripts\python.exe -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
.venv-yolo\Scripts\python.exe -m pip install -r requirements-yolov7.txt
```

These commands install CPU PyTorch wheels. For a compatible NVIDIA environment, use the matching PyTorch CUDA wheel index; the historical experiment used CUDA 11.8. A Windows AMD graphics card does not provide CUDA. CPU runs are suitable for small checks; training is better suited to a compatible GPU.

On macOS/Linux, create the environment with `python3.10 -m venv .venv-yolo` and use `.venv-yolo/bin/python` instead of the Windows executable.

## Check inputs

The toolkit expects an existing, trusted `.pt` checkpoint. Put initial weights at `weights/yolov7.pt`, or use an existing trained checkpoint for evaluation. Checkpoints are not included and can contain executable serialized objects, so use a trusted source.

Configure the real dataset and class map, then validate it as described in the root README. If labels change between runs, remove the corresponding generated YOLO `.cache` files to avoid stale label metadata.

## Prepare a run

This command validates the inputs and writes the run files without starting training:

```powershell
.venv\Scripts\python.exe -m robosubvision yolo --task train --data configs/robosub.yaml --groups data/robosub/groups.csv --repo external/yolov7 --weights weights/yolov7.pt --python .venv-yolo/Scripts/python.exe --output runs/train-plan --device cpu --epochs 50 --batch-size 8 --dry-run
```

Inspect `run_manifest.json` and `dataset_validation.json`. The prepared YAML uses absolute paths so running YOLO from a different working directory does not change which files are loaded.

## Train

Remove `--dry-run` and use a fresh output directory:

```powershell
.venv\Scripts\python.exe -m robosubvision yolo --task train --data configs/robosub.yaml --groups data/robosub/groups.csv --repo external/yolov7 --weights weights/yolov7.pt --python .venv-yolo/Scripts/python.exe --output runs/train-01 --device cpu --epochs 50 --batch-size 8
```

Use `--device 0` with a working NVIDIA environment. Training output goes to `runs/train-01/train/`; logs are written to `runs/train-01/console.log`.

## Evaluate

```powershell
.venv\Scripts\python.exe -m robosubvision yolo --task val --data configs/robosub.yaml --groups data/robosub/groups.csv --repo external/yolov7 --weights runs/train-01/train/weights/best.pt --python .venv-yolo/Scripts/python.exe --output runs/eval-01 --device cpu
```

Use `--task test` for a configured independent test split. Retain the upstream evaluation log and its settings with the result.

For annotated inference images, use `--task detect --source PATH_TO_IMAGES` with the remaining run arguments. Prediction labels include confidence scores.

The separate detector environment and real-data training still require a complete run check; see [test status](TESTING.md).
