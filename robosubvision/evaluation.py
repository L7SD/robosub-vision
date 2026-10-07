from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from .dataset import Dataset
from .labels import Box, intersection_over_union, label_path, read_boxes


def match_detections(
    truth: list[Box],
    predictions: list[Box],
    class_count: int,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.25,
) -> list[dict]:
    if not (0 < iou_threshold <= 1 and 0 <= confidence_threshold <= 1):
        raise ValueError("Invalid IoU or confidence threshold")
    counts = [{"tp": 0, "fp": 0, "fn": 0} for _ in range(class_count)]
    unmatched = set(range(len(truth)))
    for prediction in sorted(predictions, key=lambda b: -b.confidence):
        if prediction.confidence < confidence_threshold:
            continue
        candidates = [i for i in unmatched if truth[i].class_id == prediction.class_id]
        best = max(
            candidates,
            key=lambda i: intersection_over_union(prediction, truth[i]),
            default=None,
        )
        if (
            best is not None
            and intersection_over_union(prediction, truth[best]) >= iou_threshold
        ):
            counts[prediction.class_id]["tp"] += 1
            unmatched.remove(best)
        else:
            counts[prediction.class_id]["fp"] += 1
    for index in unmatched:
        counts[truth[index].class_id]["fn"] += 1
    return counts


def summarize(counts: dict) -> dict:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
    }


def evaluate(
    dataset: Dataset,
    predictions_dir: Path,
    split: str = "val",
    *,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.25,
) -> dict:
    if split not in dataset.splits or not predictions_dir.is_dir():
        raise ValueError("Requested split or predictions directory does not exist")
    images = dataset.splits[split]
    expected = {p.stem for p in images}
    extras = {p.stem for p in predictions_dir.glob("*.txt")} - expected
    if extras:
        raise ValueError(
            f"Prediction files do not belong to this split: {sorted(extras)}"
        )
    totals = [{"tp": 0, "fp": 0, "fn": 0} for _ in dataset.names]
    missing = 0
    for image in images:
        truth = read_boxes(label_path(image), len(dataset.names))
        path = predictions_dir / f"{image.stem}.txt"
        predictions = (
            read_boxes(path, len(dataset.names), predictions=True)
            if path.exists()
            else []
        )
        missing += not path.exists()
        counts = match_detections(
            truth, predictions, len(dataset.names), iou_threshold, confidence_threshold
        )
        for total, row in zip(totals, counts):
            for key in total:
                total[key] += row[key]
    return {
        "metric": "fixed-threshold class-aware one-to-one detection matching (not mAP)",
        "split": split,
        "images": len(images),
        "iou_threshold": iou_threshold,
        "confidence_threshold": confidence_threshold,
        "images_without_prediction_file": missing,
        "micro": summarize(
            {key: sum(c[key] for c in totals) for key in ["tp", "fp", "fn"]}
        ),
        "per_class": {name: summarize(row) for name, row in zip(dataset.names, totals)},
        "note": "Missing prediction files are counted as zero detections.",
    }


def overlay(
    image: Path, names: list[str], output: Path, predictions: Path | None = None
) -> None:
    truth = read_boxes(label_path(image), len(names))
    predicted = (
        read_boxes(predictions, len(names), predictions=True) if predictions else []
    )
    with Image.open(image) as source:
        canvas = ImageOps.exif_transpose(source).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for boxes, color, prefix in [
        (truth, "#20b46a", "GT"),
        (predicted, "#ff653f", "PRED"),
    ]:
        for box in boxes:
            x1, y1, x2, y2 = box.corners
            coords = (
                x1 * canvas.width,
                y1 * canvas.height,
                x2 * canvas.width,
                y2 * canvas.height,
            )
            draw.rectangle(coords, outline=color, width=3)
            draw.text(
                (coords[0] + 2, max(0, coords[1] - 13)),
                f"{prefix}: {names[box.class_id]}",
                fill=color,
            )
    if output.exists():
        raise ValueError("Overlay output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
