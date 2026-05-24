# CULane Evaluation — 3295 frames sampled across 3 driver splits
_stride 30, pixel-level protocol on the binary lane-line mask_

| Metric | Value |
| :--- | ---: |
| Pixel Precision | 0.4739 |
| Pixel Recall    | 0.3626 |
| Pixel F1        | 0.4108 |
| Pixel IoU       | 0.2585 |
| Mean Latency    | 51.27 ms |

## Per-driver breakdown

| Driver split | Frames | Precision | Recall | F1 | IoU |
| :--- | ---: | ---: | ---: | ---: | ---: |
| driver_23_30frame | 2089 | 0.4792 | 0.3775 | 0.4223 | 0.2677 |
| driver_161_90frame | 612 | 0.4963 | 0.3871 | 0.4350 | 0.2779 |
| driver_182_30frame | 594 | 0.4311 | 0.2896 | 0.3465 | 0.2095 |