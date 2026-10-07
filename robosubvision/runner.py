"""Run a local YOLOv7 checkout and record its inputs and configuration."""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import yaml
from .dataset import load_dataset, validate_dataset

YOLOV7_COMMIT = "a207844b1ce82d204ab36d87d496728d3d2348e7"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_command(
    *,
    repo: Path,
    python: str,
    task: str,
    data: Path,
    weights: Path,
    output: Path,
    image_size: int = 640,
    batch_size: int = 8,
    epochs: int = 50,
    device: str = "cpu",
    source: Path | None = None,
) -> list[str]:
    if task not in {"train", "val", "test", "detect"}:
        raise ValueError("Invalid YOLO task")
    if image_size < 32 or image_size % 32 or batch_size < 1 or epochs < 1:
        raise ValueError(
            "Image size must be a positive multiple of 32; batch size/epochs must be positive"
        )
    script = (
        "train.py"
        if task == "train"
        else "detect.py" if task == "detect" else "test.py"
    )
    command = [
        python,
        str(repo / script),
        "--weights",
        str(weights),
        "--img-size",
        str(image_size),
        "--device",
        device,
        "--project",
        str(output),
        "--name",
        task,
    ]
    if task == "train":
        command += [
            "--data",
            str(data),
            "--batch-size",
            str(batch_size),
            "--epochs",
            str(epochs),
            "--workers",
            "0",
            "--cfg",
            str(repo / "cfg/training/yolov7.yaml"),
            "--hyp",
            str(repo / "data/hyp.scratch.p5.yaml"),
        ]
    elif task in {"val", "test"}:
        command += [
            "--data",
            str(data),
            "--task",
            task,
            "--batch-size",
            str(batch_size),
            "--save-txt",
            "--save-conf",
            "--verbose",
            "--no-trace",
        ]
    else:
        if source is None:
            raise ValueError("Detection requires --source")
        command += ["--source", str(source), "--save-txt", "--save-conf", "--no-trace"]
    return command


def run_yolo(
    *,
    config: Path,
    repo: Path,
    weights: Path,
    output: Path,
    task: str,
    groups: Path | None = None,
    python: str = sys.executable,
    execute: bool = True,
    image_size: int = 640,
    batch_size: int = 8,
    epochs: int = 50,
    device: str = "cpu",
    source: Path | None = None,
) -> dict:
    repo, weights, output = repo.resolve(), weights.resolve(), output.resolve()
    dataset = load_dataset(config)
    report = validate_dataset(dataset, groups)
    if not report["valid"]:
        raise ValueError(
            "Dataset validation failed: " + "; ".join(report["errors"][:5])
        )
    if task == "test" and "test" not in dataset.splits:
        raise ValueError("No independent test split configured")
    if not weights.is_file() or weights.suffix != ".pt":
        raise ValueError(
            "Supply an existing, trusted .pt checkpoint; weights are not bundled"
        )
    if source is not None:
        source = source.resolve()
        if not source.exists():
            raise ValueError("Detection source does not exist")
    for file in [
        "train.py",
        "test.py",
        "detect.py",
        "cfg/training/yolov7.yaml",
        "data/hyp.scratch.p5.yaml",
    ]:
        if not (repo / file).is_file():
            raise ValueError(f"Incomplete YOLOv7 checkout: {file}")
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != YOLOV7_COMMIT:
        raise ValueError(f"Use the documented YOLOv7 revision: {YOLOV7_COMMIT}")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty run directory")
    # YOLOv7 resolves manifests relative to its working directory. Make every path
    # absolute in a per-run YAML so shell location cannot silently change the data.
    output.mkdir(parents=True, exist_ok=True)
    resolved = {"nc": len(dataset.names), "names": dataset.names}
    for split, images in dataset.splits.items():
        manifest = output / f"{split}_images.txt"
        manifest.write_text("\n".join(str(p) for p in images) + "\n", encoding="utf-8")
        resolved[split] = str(manifest)
    data_file = output / "dataset.yaml"
    data_file.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    command = build_command(
        repo=repo,
        python=str(Path(python).resolve()) if Path(python).exists() else python,
        task=task,
        data=data_file,
        weights=weights,
        output=output,
        image_size=image_size,
        batch_size=batch_size,
        epochs=epochs,
        device=device,
        source=source,
    )
    manifest = {
        "status": "prepared",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_commit": revision,
        "weights_sha256": file_sha256(weights),
        "dataset_sha256": report["dataset_sha256"],
        "argv": command,
        "dataset_warnings": report["warnings"],
        "metrics_source": "Read actual upstream console.log and saved outputs; no inferred scores.",
    }
    metadata = output / "run_manifest.json"
    metadata.write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "dataset_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    if execute:
        try:
            version = subprocess.run(
                [command[0], "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                check=True,
            )
            (output / "environment.txt").write_text(version.stdout, encoding="utf-8")
            with (output / "console.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    cwd=repo,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={
                        **os.environ,
                        "WANDB_MODE": "disabled",
                        "WANDB_DISABLED": "true",
                    },
                )
            manifest["returncode"] = result.returncode
            manifest["status"] = "completed" if result.returncode == 0 else "failed"
        except (OSError, subprocess.SubprocessError):
            manifest["status"] = "failed"
            metadata.write_text(json.dumps(manifest, indent=2) + "\n")
            raise
        metadata.write_text(json.dumps(manifest, indent=2) + "\n")
        if manifest["status"] != "completed":
            raise ValueError(f"YOLOv7 failed; inspect {output / 'console.log'}")
    return manifest
