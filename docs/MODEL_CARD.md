# Model card

## Historical experiment

`notebooks/robosub_training.ipynb` records a YOLOv7 experiment with eight classes, 50 training epochs, and image size 640. The notebook uses class names `class0` through `class7`; the real label-name mapping must be restored before interpreting per-class results.

Its saved best-checkpoint validation output reports:

| Metric | Value |
|---|---:|
| Images | 192 |
| Ground-truth objects | 615 |
| Precision | 0.894 |
| Recall | 0.931 |
| mAP@0.5 | 0.932 |
| mAP@0.5:0.95 | 0.695 |

These values belong to the recorded validation run. They have not been reproduced with the current toolkit and are not independent test or field-deployment results.

## Current toolkit

The repository provides dataset validation, grouped splits, fixed-threshold detection diagnostics, and run management. Its synthetic examples exercise those tools; no new underwater checkpoint or benchmark is included.

## Reproduction inputs

- Underwater images and their bounding-box labels.
- Class names in the original label-ID order.
- The trained checkpoint, or initial weights and enough compute to retrain.
- Source-video or capture-session metadata for checking split independence.

## Evaluation gaps

Exact-image hashes cannot detect near-duplicate frames or different crops of the same scene. Session grouping is needed to assess those risks.

Further testing should cover low contrast, occlusion, reflections, motion blur, distant targets, and changes in water conditions. Latency should be measured at the intended batch size and on the intended device before making runtime claims.
