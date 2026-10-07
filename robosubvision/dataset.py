from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import random
import re
import yaml
from PIL import Image, ImageOps
from .labels import label_path, read_boxes

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Dataset:
    root: Path
    names: list[str]
    splits: dict[str, list[Path]]


def load_dataset(config: Path) -> Dataset:
    config = Path(config).resolve()
    content = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError("Dataset YAML must be a mapping")
    names = content.get("names")
    if isinstance(names, dict):
        if set(names) != set(range(len(names))):
            raise ValueError(
                "Class-map keys must be consecutive integers starting at zero"
            )
        names = [names[i] for i in range(len(names))]
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(n, str) and n.strip() for n in names)
    ):
        raise ValueError("names must contain nonempty class names in label-ID order")
    if len(set(names)) != len(names) or content.get("nc", len(names)) != len(names):
        raise ValueError("Class names must be unique and nc must match their count")
    root = (config.parent / content.get("path", ".")).resolve()
    splits = {}
    for split in ["train", "val", "test"]:
        source = content.get(split)
        if source is None:
            if split != "test":
                raise ValueError(f"Missing required split: {split}")
            continue
        sources = source if isinstance(source, list) else [source]
        images = []
        for entry in sources:
            path = (root / entry).resolve()
            if path.is_dir():
                images.extend(
                    sorted(
                        p.resolve()
                        for p in path.rglob("*")
                        if p.suffix.lower() in EXTENSIONS
                    )
                )
            elif path.is_file() and path.suffix == ".txt":
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        images.append((path.parent / line.strip()).resolve())
            else:
                raise ValueError(
                    f"Split path is not an image directory or .txt manifest: {path}"
                )
        if not images:
            raise ValueError(f"Split {split} has no images")
        if len(set(images)) != len(images):
            raise ValueError(f"Split {split} repeats image paths")
        if len({p.stem for p in images}) != len(images):
            raise ValueError(
                f"Split {split} repeats filename stems; YOLO prediction text files would collide"
            )
        splits[split] = images
    return Dataset(root, names, splits)


def load_groups(path: Path, root: Path) -> dict[Path, str]:
    groups = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"image", "group_id"} <= set(reader.fieldnames or []):
            raise ValueError("Group CSV needs image and group_id columns")
        for row in reader:
            if not row["image"].strip() or not row["group_id"].strip():
                raise ValueError("Group CSV contains blank image/group values")
            image = (root / row["image"]).resolve()
            if image in groups:
                raise ValueError(f"Duplicate group assignment for {image}")
            groups[image] = row["group_id"].strip()
    return groups


def validate_dataset(dataset: Dataset, groups_csv: Path | None = None) -> dict:
    errors, warnings, split_report = [], [], {}
    seen_hashes, seen_paths, seen_groups = {}, {}, {}
    groups = load_groups(groups_csv, dataset.root) if groups_csv else None
    digest = hashlib.sha256()
    digest.update("\n".join(dataset.names).encode())
    if groups is None:
        warnings.append(
            "No capture-session groups supplied: exact duplicates are checked, but adjacent-frame leakage is not ruled out."
        )
    if any(re.fullmatch(r"class\d+", name) for name in dataset.names):
        warnings.append(
            "Class names are placeholders. Confirm the original label-ID mapping before presenting results."
        )
    if "test" not in dataset.splits:
        warnings.append(
            "No independent test split is configured; validation metrics are for development."
        )
    for split, images in dataset.splits.items():
        counts = Counter()
        backgrounds = 0
        for image in images:
            if image in seen_paths:
                errors.append(
                    f"Image path appears in both {seen_paths[image]} and {split}: {image}"
                )
            seen_paths[image] = split
            try:
                with Image.open(image) as source:
                    rgb = ImageOps.exif_transpose(source).convert("RGB")
                    rgb.load()
                    fingerprint = hashlib.sha256(
                        str(rgb.size).encode() + rgb.tobytes()
                    ).hexdigest()
                if fingerprint in seen_hashes:
                    previous_split, previous_path = seen_hashes[fingerprint]
                    message = f"Identical decoded pixels: {previous_path} and {image}"
                    (errors if previous_split != split else warnings).append(message)
                else:
                    seen_hashes[fingerprint] = (split, image)
                labels = label_path(image)
                boxes = read_boxes(labels, len(dataset.names))
                counts.update(box.class_id for box in boxes)
                backgrounds += not boxes
                # Relative paths plus bytes allow a relocated copy to retain the same identity.
                try:
                    relative = image.relative_to(dataset.root).as_posix()
                except ValueError:
                    relative = image.as_posix()
                digest.update(f"{split}:{relative}:{fingerprint}".encode())
                digest.update(labels.read_bytes())
            except (OSError, ValueError) as error:
                errors.append(str(error))
            if groups is not None:
                group = groups.get(image)
                if group is None:
                    errors.append(f"Missing capture group: {image}")
                elif group in seen_groups and seen_groups[group] != split:
                    errors.append(
                        f"Capture group {group} occurs in both {seen_groups[group]} and {split}"
                    )
                else:
                    seen_groups[group] = split
        if not sum(counts.values()):
            errors.append(f"Split {split} contains no labeled objects")
        missing = [name for i, name in enumerate(dataset.names) if not counts[i]]
        if missing:
            warnings.append(
                f"Split {split} has no labeled examples of: {', '.join(missing)}"
            )
        split_report[split] = {
            "images": len(images),
            "boxes": sum(counts.values()),
            "background_images": backgrounds,
            "objects_per_class": {
                name: counts[i] for i, name in enumerate(dataset.names)
            },
        }
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "splits": split_report,
        "class_names": dataset.names,
        "group_check_performed": groups is not None,
        "dataset_sha256": digest.hexdigest(),
    }


def grouped_split(
    root: Path,
    groups_csv: Path,
    names: list[str],
    output: Path,
    seed: int = 42,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> Path:
    root, output = root.resolve(), output.resolve()
    groups = load_groups(groups_csv, root)
    unique = sorted(set(groups.values()))
    if len(unique) < 3 or not (
        0 < validation_fraction < 1
        and 0 < test_fraction < 1
        and validation_fraction + test_fraction < 1
    ):
        raise ValueError("Need at least three capture groups and valid split fractions")
    random.Random(seed).shuffle(unique)
    nval, ntest = max(1, round(len(unique) * validation_fraction)), max(
        1, round(len(unique) * test_fraction)
    )
    if nval + ntest >= len(unique):
        raise ValueError("Too few groups for these fractions")
    assignment = {
        g: ("val" if i < nval else "test" if i < nval + ntest else "train")
        for i, g in enumerate(unique)
    }
    for image in groups:
        if not image.is_file() or not label_path(image).is_file():
            raise ValueError(f"Image/label pair is missing: {image}")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Split output must be an empty directory")
    output.mkdir(parents=True, exist_ok=True)
    config = {"path": str(root), "nc": len(names), "names": names}
    for split in ["train", "val", "test"]:
        manifest = output / f"{split}.txt"
        paths = sorted(
            str(p) for p, group in groups.items() if assignment[group] == split
        )
        manifest.write_text("\n".join(paths) + "\n", encoding="utf-8")
        config[split] = str(manifest)
    result = output / "dataset.yaml"
    result.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with (output / "group_assignments.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["group_id", "split", "seed"])
        writer.writerows(
            (group, assignment[group], seed) for group in sorted(assignment)
        )
    return result
