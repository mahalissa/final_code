# UAVDT Evaluation — 3608 frames sampled from 70 test sequences
_IoU threshold 0.5, stride 15_

| Model | Precision | Recall | F1 | Mean Latency (ms) | TP | FP | FN |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| visdrone | 0.225 | 0.665 | 0.336 | 9.6 | 18476 | 63650 | 9305 |
| fasterrcnn | 0.213 | 0.436 | 0.286 | 45.8 | 12109 | 44854 | 15672 |

## Day vs. night breakdown

| Model | Condition | Precision | Recall | F1 | TP | FP | FN |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| visdrone | day | 0.045 | 0.711 | 0.085 | 1772 | 37298 | 722 |
| visdrone | night | 0.388 | 0.661 | 0.489 | 16704 | 26352 | 8583 |
| fasterrcnn | day | 0.041 | 0.462 | 0.075 | 1151 | 27235 | 1343 |
| fasterrcnn | night | 0.383 | 0.433 | 0.407 | 10958 | 17619 | 14329 |