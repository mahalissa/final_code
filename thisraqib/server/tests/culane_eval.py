"""
Formal Evaluation on CULane (Ground-Truth Dataset)
==================================================
Evaluates YOLOPv2's painted lane-line mask against the CULane benchmark
(Pan et al., 2018), using pixel-level precision / recall / F1 / IoU.

Coverage
--------
Walks every clip directory under each of the three CULane driver splits
that are present locally (``driver_23_30frame``, ``driver_161_90frame``,
``driver_182_30frame``). For each clip it samples one frame every
``STRIDE`` indices, capped at ``MAX_FRAMES`` after striding.

Protocol
--------
CULane labels encode the four ego-relative lane lines as integer classes
1..4 in a single-channel PNG. We collapse those four classes into a
binary "any painted-lane-line" mask, because YOLOPv2's lane head emits
a single binary lane-line mask. The predicted mask is resized to GT
resolution with nearest-neighbour interpolation when the YOLOPv2 head
returns a smaller mask.

This is a **pixel-level** evaluation, not the per-line F1 used in the
CULane leaderboard (which evaluates each lane as a 30 px-wide IoU
between predicted polylines and reference polylines). The pixel-level
metric is the closest correspondence to the mask-overlap test that
RAQIB's lane-violation logic actually uses, so it is the directly
relevant accuracy number for this project.

Metrics (aggregated over all sampled pixels)
--------------------------------------------
    * Precision = TP / (TP + FP)
    * Recall    = TP / (TP + FN)
    * F1        = 2 * P * R / (P + R)
    * IoU       = TP / (TP + FP + FN)
    * Mean per-frame inference latency (ms)

Outputs
-------
    * ``_inspect/culane_eval/culane_results.json``
    * ``_inspect/culane_eval/culane_results.md``
    * ``_inspect/culane_eval/screenshots/*.jpg`` — GT (red) + pred (cyan)

Running
-------
::

    cd server
    python -m tests.culane_eval                # full sweep
    python -m tests.culane_eval --stride 15    # denser sample, slower
    python -m tests.culane_eval --max-frames 500 --no-screenshots
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detectors.lane import LaneDetector  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("culane_eval")

# ── Configuration ────────────────────────────────────────────────────────────
CULANE_ROOT    = Path(r"L:\Misc\archive\CULane")
DRIVERS        = ("driver_23_30frame", "driver_161_90frame", "driver_182_30frame")
LABEL_ROOT     = CULANE_ROOT / "laneseg_label_w16" / "laneseg_label_w16"
OUTPUT_DIR     = ROOT / "tests" / "_inspect" / "culane_eval"
SCREENSHOT_DIR = OUTPUT_DIR / "screenshots"


# ── Frame enumeration ───────────────────────────────────────────────────────-
def harvest_frames(stride: int, max_frames: int | None
                   ) -> list[tuple[str, str, int, Path, Path]]:
    """Return ``(driver, clip, idx, img_path, ann_path)`` triples."""
    out: list[tuple[str, str, int, Path, Path]] = []
    for driver in DRIVERS:
        img_root = CULANE_ROOT / driver / driver
        ann_root = LABEL_ROOT / driver
        if not img_root.exists() or not ann_root.exists():
            logger.warning(f"Skipping {driver}: missing images or labels")
            continue
        for clip_dir in sorted(img_root.iterdir()):
            if not clip_dir.is_dir():
                continue
            ann_clip = ann_root / clip_dir.name
            if not ann_clip.exists():
                continue
            indices = sorted(int(p.stem) for p in clip_dir.glob("*.jpg"))
            for idx in indices[::stride]:
                ann = ann_clip / f"{idx:05d}.png"
                if ann.exists():
                    out.append((driver, clip_dir.name, idx,
                                clip_dir / f"{idx:05d}.jpg", ann))
    if max_frames is not None and len(out) > max_frames:
        # Stride-then-cap preserves a roughly uniform sweep across drivers.
        step = len(out) / max_frames
        out = [out[int(i * step)] for i in range(max_frames)]
    return out


# ── Screenshot helper ───────────────────────────────────────────────────────-
def draw_overlay(frame, pred_mask, gt_mask) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    if pred_mask.shape != (h, w):
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), (w, h),
                               interpolation=cv2.INTER_NEAREST)
    if gt_mask.shape != (h, w):
        gt_mask = cv2.resize(gt_mask.astype(np.uint8), (w, h),
                             interpolation=cv2.INTER_NEAREST)
    gt_layer = np.zeros_like(out)
    gt_layer[gt_mask > 0] = (0, 0, 255)         # red
    pred_layer = np.zeros_like(out)
    pred_layer[pred_mask > 0] = (255, 255, 0)   # cyan
    out = cv2.addWeighted(out, 1.0, gt_layer,   0.5, 0)
    out = cv2.addWeighted(out, 1.0, pred_layer, 0.5, 0)
    cv2.putText(out, "Red=GT  Cyan=YOLOPv2 prediction", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ── Evaluation ──────────────────────────────────────────────────────────────-
def evaluate(detector, frames, n_screenshots: int) -> dict:
    tp = fp = fn = 0
    per_driver: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "frames": 0})
    latencies: list[float] = []

    screenshot_idxs = set()
    if n_screenshots > 0 and frames:
        step = max(1, len(frames) // (n_screenshots * 3))
        screenshot_idxs = set(range(0, len(frames), step))

    saved = 0
    for i, (driver, clip, idx, img_path, ann_path) in enumerate(frames):
        frame = cv2.imread(str(img_path))
        gt_mask = cv2.imread(str(ann_path), cv2.IMREAD_GRAYSCALE)
        if frame is None or gt_mask is None:
            continue

        t0 = time.perf_counter()
        lane_data = detector.detect(frame)
        latencies.append((time.perf_counter() - t0) * 1000)

        pred_mask = lane_data["_lane_mask"]
        if pred_mask.ndim == 3:
            pred_mask = pred_mask.squeeze()
        gt_binary = (gt_mask > 0).astype(np.uint8)
        if gt_binary.ndim == 3:
            gt_binary = gt_binary.squeeze()
        if pred_mask.shape != gt_binary.shape:
            pred_mask = cv2.resize(pred_mask.astype(np.uint8),
                                   (gt_binary.shape[1], gt_binary.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        intersection = int(np.logical_and(pred_mask > 0, gt_binary > 0).sum())
        pred_pos     = int((pred_mask > 0).sum())
        gt_pos       = int((gt_binary > 0).sum())
        f_tp = intersection
        f_fp = pred_pos - intersection
        f_fn = gt_pos - intersection
        tp += f_tp; fp += f_fp; fn += f_fn
        per_driver[driver]["tp"] += f_tp
        per_driver[driver]["fp"] += f_fp
        per_driver[driver]["fn"] += f_fn
        per_driver[driver]["frames"] += 1

        if i in screenshot_idxs and saved < n_screenshots:
            out_path = SCREENSHOT_DIR / f"culane_{driver}_{clip}_{idx:05d}.jpg"
            cv2.imwrite(str(out_path), draw_overlay(frame, pred_mask, gt_binary))
            saved += 1

        if (i + 1) % 250 == 0:
            logger.info(f"  {i+1}/{len(frames)} frames")

    def _metrics(t, f_, n):
        p = t / (t + f_) if (t + f_) else 0.0
        r = t / (t + n)  if (t + n)  else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        iou = t / (t + f_ + n) if (t + f_ + n) else 0.0
        return p, r, f1, iou

    p, r, f1, iou = _metrics(tp, fp, fn)
    by_driver = {}
    for d, v in per_driver.items():
        dp, dr, df, di = _metrics(v["tp"], v["fp"], v["fn"])
        by_driver[d] = {
            "frames":    v["frames"],
            "precision": dp, "recall": dr, "f1": df, "iou": di,
            "tp": v["tp"], "fp": v["fp"], "fn": v["fn"],
        }

    return {
        "frames_evaluated":  len(latencies),
        "pixel_precision":   p,
        "pixel_recall":      r,
        "pixel_f1":          f1,
        "pixel_iou":         iou,
        "mean_latency_ms":   float(np.mean(latencies)) if latencies else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
        "by_driver":         by_driver,
    }


# ── Reporting ───────────────────────────────────────────────────────────────-
def write_outputs(results: dict, args) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    md = [
        f"# CULane Evaluation — {results['frames_evaluated']} frames sampled across "
        f"{len(results['by_driver'])} driver splits",
        f"_stride {args.stride}, pixel-level protocol on the binary lane-line mask_",
        "",
        "| Metric | Value |",
        "| :--- | ---: |",
        f"| Pixel Precision | {results['pixel_precision']:.4f} |",
        f"| Pixel Recall    | {results['pixel_recall']:.4f} |",
        f"| Pixel F1        | {results['pixel_f1']:.4f} |",
        f"| Pixel IoU       | {results['pixel_iou']:.4f} |",
        f"| Mean Latency    | {results['mean_latency_ms']:.2f} ms |",
        "",
        "## Per-driver breakdown",
        "",
        "| Driver split | Frames | Precision | Recall | F1 | IoU |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for d, v in results["by_driver"].items():
        md.append(
            f"| {d} | {v['frames']} | {v['precision']:.4f} | {v['recall']:.4f} "
            f"| {v['f1']:.4f} | {v['iou']:.4f} |"
        )

    (OUTPUT_DIR / "culane_results.md").write_text("\n".join(md), encoding="utf-8")
    with open(OUTPUT_DIR / "culane_results.json", "w", encoding="utf-8") as f:
        json.dump({"stride": args.stride, **results}, f, indent=2)
    logger.info(f"Wrote results to {OUTPUT_DIR}")


# ── CLI ─────────────────────────────────────────────────────────────────────-
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-coverage CULane evaluation harness.")
    p.add_argument("--stride", type=int, default=30,
                   help="Sample 1 frame every N (default 30 across all driver splits).")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Hard cap on total frames after striding.")
    p.add_argument("--screenshots", type=int, default=6,
                   help="Annotated frames to dump (0 to disable).")
    p.add_argument("--no-screenshots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_screenshots:
        args.screenshots = 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Harvesting frames...")
    frames = harvest_frames(args.stride, args.max_frames)
    logger.info(
        f"Harvested {len(frames)} frames from "
        f"{len({(d, c) for d, c, *_ in frames})} clips "
        f"across {len({f[0] for f in frames})} driver splits"
    )

    detector = LaneDetector()
    results = evaluate(detector, frames, args.screenshots)
    write_outputs(results, args)
    print(json.dumps({k: v for k, v in results.items()
                      if k != "by_driver"}, indent=2))


if __name__ == "__main__":
    main()
