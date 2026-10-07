"""Synthetic images and prediction fixtures for tests and examples."""

from pathlib import Path
import csv
import yaml
from PIL import Image, ImageDraw


def create_demo(output: Path) -> Path:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Demo output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    groups = []
    for split, count in [("train", 8), ("val", 4), ("test", 4)]:
        for folder in ["images", "labels"]:
            (output / split / folder).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            stem = f"{split}_{index:03d}"
            image = Image.new(
                "RGB", (256, 256), (12 + index * 7, 25 + len(split) * 10, 60)
            )
            draw = ImageDraw.Draw(image)
            cls = index % 2
            x, y = 70 + index * 6, 95 + index * 3
            bounds = (x - 25, y - 30, x + 25, y + 30)
            if cls == 0:
                draw.rectangle(bounds, fill="#ffbc42")
            else:
                draw.ellipse(bounds, fill="#a5e2e5")
            relative = Path(split) / "images" / f"{stem}.png"
            image.save(output / relative)
            label = f"{cls} {x/256:.8f} {y/256:.8f} {50/256:.8f} {60/256:.8f}"
            (output / split / "labels" / f"{stem}.txt").write_text(label + "\n")
            groups.append((relative.as_posix(), f"{split}_capture_{index // 2}"))
            if split == "val":
                predictions = output / "fixture_predictions"
                predictions.mkdir(exist_ok=True)
                # Include one missed object and one duplicate detection.
                text = "" if index == 2 else label + " 0.9\n"
                if index == 1:
                    text += label + " 0.6\n"
                (predictions / f"{stem}.txt").write_text(text)
    with (output / "groups.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "group_id"])
        writer.writerows(groups)
    config = {
        "path": ".",
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 2,
        "names": ["synthetic_rectangle", "synthetic_ellipse"],
    }
    path = output / "dataset.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "SYNTHETIC_DATA.txt").write_text(
        "Generated geometry fixtures only. No detector was trained or run to produce fixture_predictions.\n"
    )
    return path
