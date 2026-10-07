# Tests

Run the core suite from the project root in the Python 3.12 environment:

```sh
python -m unittest discover -s tests -v
```

All 9 tests passed. They cover:

- Valid synthetic fixtures and image overlays.
- Malformed boxes, missing labels, invalid classes, and duplicate ground-truth boxes.
- Identical images and capture groups appearing across splits.
- Grouped split completeness and separation.
- Class-aware, one-to-one detection matching and confidence filtering.
- Prediction files that do not belong to the evaluated split.
- YOLOv7 command construction with paths containing spaces.

The synthetic evaluation produces 3 true positives, 1 false positive, and 1 false negative. It checks matching and reporting, not detector accuracy.

The separate YOLOv7 environment has not completed a training or inference run with this toolkit. Reproduction requires the underwater dataset, its class-name mapping, and trusted weights. The saved notebook documents a historical run; its metrics have not been reproduced here.

GitHub Actions runs the core tests without a GPU or detector installation.
