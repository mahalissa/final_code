"""
Formal Evaluation on UAVDT (Ground-Truth Dataset)
=================================================
Computes absolute accuracy metrics for the vehicle detectors against
human-annotated bounding boxes from the UAVDT benchmark (Du et al., 2018).

Coverage
--------
Walks every sequence in ``DATASET_ROOT/test/img`` and samples one frame
per ``STRIDE`` indices, capped at ``MAX_FRAMES`` (default unlimited).
With the default stride this exercises every sequence in the test split
rather than the previous two-sequence sample.

Matching protocol
-----------------
For each frame, predicted boxes are greedily matched to the highest-IoU
unmatched ground-truth box **of the same class**. A pairing with
IoU >= IOU_THRESHOLD (default 0.5) counts as a true positive; otherwise
the prediction is a false positive. Any unmatched ground-truth box is a
false negative.

Metrics reported (per model, overall and per-sequence)
------------------------------------------------------
    * Precision    = TP / (TP + FP)
    * Recall       = TP / (TP + FN)
    * F1 score     = 2 * P * R / (P + R)
    * Mean latency = average per-frame ``detector.detect`` time (ms)
    * Raw TP / FP / FN counts

Class mapping
-------------
    * ``car``     -> ``car``
    * ``truck``   -> ``truck``
    * ``bus``     -> ``bus``
    * ``vehicle`` -> ``car`` (generic UAVDT fallback)

Outputs
-------
    * ``_inspect/uavdt_eval/uavdt_results.json`` — overall + per-sequence
    * ``_inspect/uavdt_eval/uavdt_results.md``   — markdown summary table
    * ``_inspect/uavdt_eval/screenshots/*.jpg``  — annotated sample frames

Running
-------
::

    cd server
    python -m tests.uavdt_eval                # full sweep, default stride
    python -m tests.uavdt_eval --stride 30    # sparser sample, faster
    python -m tests.uavdt_eval --max-frames 1000 --no-screenshots
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Add the ``server`` package to sys.path so this file can be invoked either
# as ``python -m tests.uavdt_eval`` or directly as a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detectors.registry import DetectorRegistry  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("uavdt_eval")

# ── Configuration ────────────────────────────────────────────────────────────
DATASET_ROOT   = Path(r"L:\Misc\uavdt-DatasetNinja")
TEST_IMG_DIR   = DATASET_ROOT / "test" / "img"
TEST_ANN_DIR   = DATASET_ROOT / "test" / "ann"
OUTPUT_DIR     = ROOT / "tests" / "_inspect" / "uavdt_eval"
SCREENSHOT_DIR = OUTPUT_DIR / "screenshots"

IOU_THRESHOLD  = 0.5

# UAVDT sequence id is the prefix before ``_imgNNNNNN.jpg``.
FRAME_RE = re.compile(r"^(?P<seq>[A-Z]\d+)_img(?P<idx>\d+)\.jpg$")

CLASS_MAP = {
    "car":     "car",
    "truck":   "truck",
    "bus":     "bus",
    "vehicle": "car",
}
VEHICLE_LABELS = ("car", "truck", "bus")

# Sequences whose prefix starts with ``M`` are night-time in the official
# UAVDT split; ``S`` sequences are daytime. The split is used only for the
# per-condition breakdown in the report.
NIGHT_PREFIX = "M"


# ── Geometry helpers ────────────────────────────────────────────────────────-
def iou(box_a, box_b) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    area_a = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    area_b = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_ground_truth(seq: str, idx: int) -> list[dict] | None:
    """Read the DatasetNinja annotation JSON for a UAVDT frame.

    Returns ``None`` if the annotation file does not exist (some
    sequences have small gaps).
    """
    ann_path = TEST_ANN_DIR / f"{seq}_img{idx:06d}.jpg.json"
    if not ann_path.exists():
        return None
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    objs = []
    for obj in data.get("objects", []):
        label = CLASS_MAP.get(obj["classTitle"])
        if not label:
            continue
        pts = obj["points"]["exterior"]
        objs.append({
            "label": label,
            "box":   [pts[0][0], pts[0][1], pts[1][0], pts[1][1]],
        })
    return objs


# ── Frame enumeration ───────────────────────────────────────────────────────-
def harvest_frames(stride: int, max_frames: int | None) -> list[tuple[str, int, Path]]:
    """Return ``(seq, idx, path)`` triples covering every annotated frame
    at the requested ``stride``, grouped by sequence."""
    by_seq: dict[str, list[int]] = defaultdict(list)
    for p in TEST_IMG_DIR.iterdir():
        m = FRAME_RE.match(p.name)
        if m:
            by_seq[m.group("seq")].append(int(m.group("idx")))

    frames: list[tuple[str, int, Path]] = []
    for seq in sorted(by_seq):
        for idx in sorted(by_seq[seq])[::stride]:
            frames.append((seq, idx, TEST_IMG_DIR / f"{seq}_img{idx:06d}.jpg"))

    if max_frames is not None and len(frames) > max_frames:
        random.Random(0).shuffle(frames)
        frames = frames[:max_frames]
        frames.sort(key=lambda x: (x[0], x[1]))

    return frames


# ── Per-frame evaluation ────────────────────────────────────────────────────-
def score_frame(preds, gt) -> tuple[int, int, int, list[tuple[list[int], str]]]:
    """Greedy IoU matching; returns (tp, fp, fn, fp_boxes).

    ``fp_boxes`` is the list of predicted boxes that did not match any
    GT — used for the annotated screenshots.
    """
    matched = set()
    tp = fp = 0
    fp_boxes: list[tuple[list[int], str]] = []
    for p in preds:
        best_iou, best_idx = 0.0, -1
        for i, g in enumerate(gt):
            if i in matched or g["label"] != p["label"]:
                continue
            v = iou(p["box"], g["box"])
            if v > best_iou:
                best_iou, best_idx = v, i
        if best_iou >= IOU_THRESHOLD:
            tp += 1
            matched.add(best_idx)
        else:
            fp += 1
            fp_boxes.append((p["box"], p["label"]))
    fn = len(gt) - len(matched)
    return tp, fp, fn, fp_boxes


# ── Annotated screenshots ───────────────────────────────────────────────────-
def draw_annotated(frame, gt, preds, fp_boxes) -> np.ndarray:
    out = frame.copy()
    for g in gt:
        x1, y1, x2, y2 = [int(v) for v in g["box"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 200, 0), 1)  # blue = GT
    for p in preds:
        x1, y1, x2, y2 = [int(v) for v in p["box"]]
        is_fp = (p["box"], p["label"]) in fp_boxes
        colour = (0, 0, 255) if is_fp else (0, 255, 0)              # red FP, green TP
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(out, p["label"], (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
    cv2.putText(out, "Blue=GT  Green=TP  Red=FP", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def evaluate(name, detector, frames, n_screenshots: int) -> dict:
    """Run one detector across ``frames`` and return aggregate metrics."""
    logger.info(f"Evaluating {name} on {len(frames)} frames")
    totals = {"tp": 0, "fp": 0, "fn": 0}
    per_seq: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    latencies: list[float] = []

    # Choose frames to screenshot in advance: spread across the sweep,
    # then prefer ones with non-empty ground truth so they are illustrative.
    screenshot_idxs = set()
    if n_screenshots > 0 and frames:
        step = max(1, len(frames) // (n_screenshots * 3))
        screenshot_idxs = set(range(0, len(frames), step))

    saved = 0
    for i, (seq, idx, img_path) in enumerate(frames):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        gt = load_ground_truth(seq, idx)
        if gt is None:
            continue

        t0 = time.perf_counter()
        preds = detector.detect(frame)
        latencies.append((time.perf_counter() - t0) * 1000)

        preds = [p for p in preds if p["label"] in VEHICLE_LABELS]
        tp, fp, fn, fp_boxes = score_frame(preds, gt)
        totals["tp"] += tp; totals["fp"] += fp; totals["fn"] += fn
        per_seq[seq]["tp"] += tp; per_seq[seq]["fp"] += fp; per_seq[seq]["fn"] += fn

        if i in screenshot_idxs and saved < n_screenshots and gt:
            out_path = SCREENSHOT_DIR / f"{name}_{seq}_{idx:06d}.jpg"
            cv2.imwrite(str(out_path), draw_annotated(frame, gt, preds, fp_boxes))
            saved += 1

        if (i + 1) % 250 == 0:
            logger.info(f"  {name}: {i+1}/{len(frames)} frames")

    def _metrics(d):
        tp, fp, fn = d["tp"], d["fp"], d["fn"]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f

    p, r, f = _metrics(totals)

    # Day/night breakdown by sequence prefix.
    cond_totals = {"day": {"tp": 0, "fp": 0, "fn": 0}, "night": {"tp": 0, "fp": 0, "fn": 0}}
    for seq, d in per_seq.items():
        bucket = "night" if seq.startswith(NIGHT_PREFIX) else "day"
        for k in ("tp", "fp", "fn"):
            cond_totals[bucket][k] += d[k]
    cond_metrics = {}
    for bucket, d in cond_totals.items():
        cp, cr, cf = _metrics(d)
        cond_metrics[bucket] = {"precision": cp, "recall": cr, "f1": cf, **d}

    return {
        "model":            name,
        "precision":        p,
        "recall":           r,
        "f1":               f,
        "mean_latency_ms":  float(np.mean(latencies)) if latencies else 0.0,
        "frames_scored":    sum(1 for _ in latencies),
        **totals,
        "by_condition":     cond_metrics,
        "per_sequence":     {
            seq: {**d, **dict(zip(("precision", "recall", "f1"), _metrics(d)))}
            for seq, d in per_seq.items()
        },
    }


# ── Reporting ───────────────────────────────────────────────────────────────-
def write_outputs(results: list[dict], frames_count: int, args) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    md = [
        f"# UAVDT Evaluation — {frames_count} frames sampled from 70 test sequences",
        f"_IoU threshold {IOU_THRESHOLD}, stride {args.stride}_",
        "",
        "| Model | Precision | Recall | F1 | Mean Latency (ms) | TP | FP | FN |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        md.append(
            f"| {r['model']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} "
            f"| {r['mean_latency_ms']:.1f} | {r['tp']} | {r['fp']} | {r['fn']} |"
        )
    md += ["", "## Day vs. night breakdown", "",
           "| Model | Condition | Precision | Recall | F1 | TP | FP | FN |",
           "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in results:
        for cond, d in r["by_condition"].items():
            md.append(
                f"| {r['model']} | {cond} | {d['precision']:.3f} | {d['recall']:.3f} "
                f"| {d['f1']:.3f} | {d['tp']} | {d['fp']} | {d['fn']} |"
            )

    (OUTPUT_DIR / "uavdt_results.md").write_text("\n".join(md), encoding="utf-8")
    with open(OUTPUT_DIR / "uavdt_results.json", "w", encoding="utf-8") as f:
        json.dump({"frames": frames_count, "stride": args.stride,
                   "iou_threshold": IOU_THRESHOLD, "results": results},
                  f, indent=2)
    logger.info(f"Wrote results to {OUTPUT_DIR}")


# ── CLI ─────────────────────────────────────────────────────────────────────-
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-coverage UAVDT evaluation harness.")
    p.add_argument("--stride", type=int, default=15,
                   help="Sample 1 frame every N (default 15, ~3.6k frames).")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Hard cap on total frames after striding.")
    p.add_argument("--models", nargs="+", default=["visdrone", "fasterrcnn"],
                   help="Detectors to evaluate (registry keys).")
    p.add_argument("--screenshots", type=int, default=6,
                   help="Annotated frames to dump per model (0 to disable).")
    p.add_argument("--no-screenshots", action="store_true",
                   help="Disable screenshot writing (overrides --screenshots).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_screenshots:
        args.screenshots = 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Harvesting frames...")
    frames = harvest_frames(args.stride, args.max_frames)
    logger.info(f"Harvested {len(frames)} frames across {len({f[0] for f in frames})} sequences")

    registry = DetectorRegistry()
    results = []
    for name in args.models:
        det = registry.get_vehicle(name)
        results.append(evaluate(name, det, frames, args.screenshots))

    write_outputs(results, len(frames), args)


if __name__ == "__main__":
    main()
