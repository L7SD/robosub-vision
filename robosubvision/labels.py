from dataclasses import dataclass
from pathlib import Path
import math


@dataclass(frozen=True)
class Box:
    class_id: int
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0

    @property
    def corners(self):
        return (
            self.x - self.width / 2,
            self.y - self.height / 2,
            self.x + self.width / 2,
            self.y + self.height / 2,
        )


def read_boxes(path: Path, class_count: int, *, predictions: bool = False) -> list[Box]:
    boxes = []
    seen = set()
    for number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue  # Empty label files are explicitly annotated background images.
        fields = line.split()
        expected = 6 if predictions else 5
        try:
            values = list(map(float, fields))
            if len(values) != expected or not all(math.isfinite(v) for v in values):
                raise ValueError("wrong field count or nonfinite value")
            cls, x, y, width, height = values[:5]
            if not cls.is_integer() or not 0 <= cls < class_count:
                raise ValueError("class ID is outside the configured class map")
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError(
                    "coordinates must be normalized; dimensions must be positive"
                )
            confidence = values[5] if predictions else 1.0
            if not 0 <= confidence <= 1:
                raise ValueError("confidence must be within [0, 1]")
            box = Box(int(cls), x, y, width, height, confidence)
            if min(box.corners[:2]) < -1e-5 or max(box.corners[2:]) > 1 + 1e-5:
                raise ValueError("box extends outside the image")
            if not predictions and box in seen:
                raise ValueError("duplicate ground-truth box")
            seen.add(box)
            boxes.append(box)
        except ValueError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    return boxes


def label_path(image: Path) -> Path:
    parts = list(image.parts)
    indices = [index for index, part in enumerate(parts) if part == "images"]
    if not indices:
        raise ValueError(f"Image path must contain an images directory: {image}")
    parts[indices[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def intersection_over_union(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a.corners
    bx1, by1, bx2, by2 = b.corners
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0 else 0.0
