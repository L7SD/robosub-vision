import argparse
import json
from pathlib import Path
import subprocess
import sys
from .dataset import grouped_split, load_dataset, validate_dataset
from .demo import create_demo
from .evaluation import evaluate, overlay
from .runner import run_yolo


def emit(report: dict, output: Path | None = None):
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if output:
        if output.exists():
            raise ValueError("Output file already exists; choose a new path")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


def main():
    parser = argparse.ArgumentParser(
        description="RoboSub vision data and evaluation toolkit"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser(
        "demo", help="Create geometry fixtures and synthetic predictions"
    )
    demo.add_argument("--output", type=Path, required=True)
    check = commands.add_parser(
        "validate", help="Validate labels, image duplicates and optional capture groups"
    )
    check.add_argument("--data", type=Path, required=True)
    check.add_argument("--groups", type=Path)
    check.add_argument("--output", type=Path)
    split = commands.add_parser(
        "split", help="Split capture groups, never individual frames"
    )
    split.add_argument("--root", type=Path, required=True)
    split.add_argument("--groups", type=Path, required=True)
    split.add_argument(
        "--names",
        type=Path,
        required=True,
        help="Text file: one class name per line in ID order",
    )
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--seed", type=int, default=42)
    diagnostic = commands.add_parser(
        "evaluate", help="Fixed-threshold diagnostics; use YOLO test.py for mAP"
    )
    diagnostic.add_argument("--data", type=Path, required=True)
    diagnostic.add_argument("--predictions", type=Path, required=True)
    diagnostic.add_argument("--split", choices=["train", "val", "test"], default="val")
    diagnostic.add_argument("--iou", type=float, default=0.5)
    diagnostic.add_argument("--confidence", type=float, default=0.25)
    diagnostic.add_argument("--output", type=Path)
    visual = commands.add_parser(
        "overlay", help="Draw ground-truth and optional predicted boxes"
    )
    visual.add_argument("--data", type=Path, required=True)
    visual.add_argument("--image", type=Path, required=True)
    visual.add_argument("--predictions", type=Path)
    visual.add_argument("--output", type=Path, required=True)
    run = commands.add_parser(
        "yolo", help="Run a pinned, separately installed YOLOv7 checkout"
    )
    run.add_argument(
        "--task", choices=["train", "val", "test", "detect"], required=True
    )
    for flag in ["data", "repo", "weights", "output"]:
        run.add_argument(f"--{flag}", type=Path, required=True)
    run.add_argument("--groups", type=Path)
    run.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable of the separate YOLO environment",
    )
    run.add_argument("--source", type=Path)
    run.add_argument("--device", default="cpu")
    run.add_argument("--image-size", type=int, default=640)
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--epochs", type=int, default=50)
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Write a run plan without executing the detector",
    )
    args = parser.parse_args()
    try:
        if args.command == "demo":
            print(create_demo(args.output))
        elif args.command == "validate":
            report = validate_dataset(load_dataset(args.data), args.groups)
            emit(report, args.output)
            if not report["valid"]:
                return 1
        elif args.command == "split":
            names = [
                line.strip()
                for line in args.names.read_text().splitlines()
                if line.strip()
            ]
            config = grouped_split(
                args.root, args.groups, names, args.output, args.seed
            )
            report = validate_dataset(load_dataset(config), args.groups)
            emit(report, args.output / "validation.json")
            if not report["valid"]:
                return 1
        elif args.command == "evaluate":
            dataset = load_dataset(args.data)
            report = validate_dataset(dataset)
            if not report["valid"]:
                raise ValueError("Dataset is invalid; run validate to inspect errors")
            emit(
                evaluate(
                    dataset,
                    args.predictions,
                    args.split,
                    iou_threshold=args.iou,
                    confidence_threshold=args.confidence,
                ),
                args.output,
            )
        elif args.command == "overlay":
            overlay(
                args.image, load_dataset(args.data).names, args.output, args.predictions
            )
            print(args.output)
        else:
            report = run_yolo(
                config=args.data,
                repo=args.repo,
                weights=args.weights,
                output=args.output,
                task=args.task,
                groups=args.groups,
                python=args.python,
                execute=not args.dry_run,
                image_size=args.image_size,
                batch_size=args.batch_size,
                epochs=args.epochs,
                device=args.device,
                source=args.source,
            )
            emit(report)
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
