import csv
from pathlib import Path
import shutil
import tempfile
import unittest
from robosubvision.dataset import grouped_split, load_dataset, validate_dataset
from robosubvision.demo import create_demo
from robosubvision.evaluation import evaluate, match_detections, overlay
from robosubvision.labels import Box, intersection_over_union, read_boxes
from robosubvision.runner import build_command


class ToolkitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = create_demo(self.root / "demo")
        self.dataset = load_dataset(self.config)

    def test_valid_demo_and_explicit_synthetic_diagnostics(self):
        report = validate_dataset(self.dataset, self.root / "demo/groups.csv")
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["splits"]["train"]["images"], 8)
        scores = evaluate(self.dataset, self.root / "demo/fixture_predictions")
        self.assertEqual(scores["micro"]["tp"], 3)
        self.assertEqual(scores["micro"]["fp"], 1)
        self.assertEqual(scores["micro"]["fn"], 1)
        self.assertAlmostEqual(scores["micro"]["precision"], 0.75)
        overlay(
            self.root / "demo/val/images/val_001.png",
            self.dataset.names,
            self.root / "overlay.png",
            self.root / "demo/fixture_predictions/val_001.txt",
        )
        self.assertTrue((self.root / "overlay.png").is_file())

    def test_cross_split_duplicate_image_is_rejected(self):
        shutil.copyfile(
            self.root / "demo/train/images/train_000.png",
            self.root / "demo/val/images/val_000.png",
        )
        report = validate_dataset(self.dataset)
        self.assertFalse(report["valid"])
        self.assertTrue(
            any("Identical decoded pixels" in error for error in report["errors"])
        )

    def test_missing_label_and_out_of_range_class_fail(self):
        label = self.root / "demo/train/labels/train_000.txt"
        label.unlink()
        self.assertFalse(validate_dataset(self.dataset)["valid"])
        label.write_text("8 0.5 0.5 0.2 0.2\n")
        self.assertFalse(validate_dataset(self.dataset)["valid"])

    def test_bad_coordinates_nan_and_duplicate_ground_truth_fail(self):
        path = self.root / "label.txt"
        for text in [
            "0 nan .5 .1 .1",
            "0 .5 .5 0 .1",
            "0 .1 .1 .8 .8",
            "0.5 .5 .5 .1 .1",
            "0 .5 .5 .1 .1\n0 .5 .5 .1 .1",
        ]:
            path.write_text(text)
            with self.assertRaises(ValueError):
                read_boxes(path, 2)

    def test_capture_groups_cannot_cross_splits(self):
        path = self.root / "demo/groups.csv"
        with path.open(newline="") as source:
            rows = list(csv.DictReader(source))
        rows[8]["group_id"] = rows[0]["group_id"]
        with path.open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=["image", "group_id"])
            writer.writeheader()
            writer.writerows(rows)
        report = validate_dataset(self.dataset, path)
        self.assertFalse(report["valid"])
        self.assertTrue(any("Capture group" in e for e in report["errors"]))

    def test_group_split_has_no_group_overlap(self):
        path = grouped_split(
            self.root / "demo",
            self.root / "demo/groups.csv",
            self.dataset.names,
            self.root / "splits",
        )
        result = validate_dataset(load_dataset(path), self.root / "demo/groups.csv")
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(sum(s["images"] for s in result["splits"].values()), 16)

    def test_iou_matching_is_class_aware_and_one_to_one(self):
        truth = Box(0, 0.5, 0.5, 0.2, 0.2)
        wrong = Box(1, 0.5, 0.5, 0.2, 0.2, 0.9)
        self.assertAlmostEqual(intersection_over_union(truth, truth), 1.0)
        result = match_detections([truth], [wrong, truth, truth], 2)
        self.assertEqual(
            result, [{"tp": 1, "fp": 1, "fn": 0}, {"tp": 0, "fp": 1, "fn": 0}]
        )
        result = match_detections([truth], [Box(0, 0.5, 0.5, 0.2, 0.2, 0.1)], 2)
        self.assertEqual(result[0]["fn"], 1)

    def test_foreign_predictions_are_rejected(self):
        path = self.root / "demo/fixture_predictions/unknown.txt"
        path.write_text("")
        with self.assertRaisesRegex(ValueError, "do not belong"):
            evaluate(self.dataset, path.parent)

    def test_upstream_arguments_keep_paths_as_individual_arguments(self):
        command = build_command(
            repo=Path("a path/yolov7"),
            python="python",
            task="test",
            data=Path("a path/data.yaml"),
            weights=Path("weights.pt"),
            output=Path("run path"),
        )
        self.assertIn("a path/data.yaml", command)
        self.assertIn("--save-conf", command)
        self.assertEqual(command[command.index("--task") + 1], "test")
        with self.assertRaises(ValueError):
            build_command(
                repo=Path("repo"),
                python="python",
                task="train",
                data=self.config,
                weights=Path("weights.pt"),
                output=self.root,
                image_size=641,
            )


if __name__ == "__main__":
    unittest.main()
