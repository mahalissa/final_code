"""
Formal Evaluation on CULane (Ground-Truth Dataset)
=================================================
Evaluates YOLOPv2 lane segmentation against the CULane benchmark dataset.

Metrics:
  • IoU (Intersection over Union)
  • F1 Score (Pixel-level)
  • Mean Latency (ms)

Dataset structure used:
  • Images: L:/Misc/archive/CULane/driver_23_30frame/driver_23_30frame/05151640_0419.MP4/*.jpg
  • Labels: L:/Misc/archive/CULane/laneseg_label_w16/laneseg_label_w16/driver_23_30frame/05151640_0419.MP4/*.png
"""
import os
import sys
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch

# Add server to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detectors.lane import LaneDetector

# Configuration
CULANE_ROOT = Path(r"L:\Misc\archive\CULane")
IMG_BASE = CULANE_ROOT / "driver_23_30frame" / "driver_23_30frame"
ANN_BASE = CULANE_ROOT / "laneseg_label_w16" / "laneseg_label_w16" / "driver_23_30frame"
OUTPUT_DIR = ROOT / "tests" / "_inspect" / "culane_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sequence to evaluate
SEQ = "05151640_0419.MP4"
NUM_FRAMES = 50 # Evaluate 50 frames

def main():
    detector = LaneDetector()
    
    tp_pixels = 0
    fp_pixels = 0
    fn_pixels = 0
    latencies = []
    
    img_dir = IMG_BASE / SEQ
    ann_dir = ANN_BASE / SEQ
    
    # CULane images are named 00000.jpg, 00030.jpg, etc.
    frame_indices = sorted([int(f.stem) for f in img_dir.glob("*.jpg")])[:NUM_FRAMES]
    
    print(f"Evaluating YOLOPv2 on {len(frame_indices)} frames from {SEQ}...")
    
    for idx in frame_indices:
        img_path = img_dir / f"{idx:05d}.jpg"
        ann_path = ann_dir / f"{idx:05d}.png"
        
        if not ann_path.exists():
            continue
            
        frame = cv2.imread(str(img_path))
        gt_mask = cv2.imread(str(ann_path), cv2.IMREAD_GRAYSCALE)
        
        if frame is None or gt_mask is None:
            continue
            
        # Run inference
        t0 = time.perf_counter()
        lane_data = detector.detect(frame)
        latencies.append((time.perf_counter() - t0) * 1000)
        
        # YOLOPv2 mask is 0/1 (internal cached _lane_mask)
        # Note: YOLOPv2 uses 1280x720 internal processing or similar, 
        # but the class recovers the original frame size.
        pred_mask = lane_data["_lane_mask"]
        
        # Binary ground truth (CULane uses 1, 2, 3, 4 for different lanes)
        gt_binary = (gt_mask > 0).astype(np.uint8)
        
        # Ensure 2D
        if pred_mask.ndim == 3:
            pred_mask = pred_mask.squeeze()
        if gt_binary.ndim == 3:
            gt_binary = gt_binary.squeeze()

        # Ensure sizes match
        if pred_mask.shape != gt_binary.shape:
            pred_mask = cv2.resize(pred_mask, (gt_binary.shape[1], gt_binary.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        # Calculate pixel-level metrics
        intersection = np.logical_and(pred_mask > 0, gt_binary > 0).sum()
        union = np.logical_or(pred_mask > 0, gt_binary > 0).sum()
        
        tp_pixels += intersection
        fp_pixels += (pred_mask > 0).sum() - intersection
        fn_pixels += (gt_binary > 0).sum() - intersection
        
    precision = tp_pixels / (tp_pixels + fp_pixels) if (tp_pixels + fp_pixels) > 0 else 0
    recall = tp_pixels / (tp_pixels + fn_pixels) if (tp_pixels + fn_pixels) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp_pixels / (tp_pixels + fp_pixels + fn_pixels) if (tp_pixels + fp_pixels + fn_pixels) > 0 else 0
    
    results = {
        "dataset": "CULane",
        "sequence": SEQ,
        "frames_evaluated": len(latencies),
        "pixel_precision": round(precision, 4),
        "pixel_recall": round(recall, 4),
        "pixel_f1": round(f1, 4),
        "pixel_iou": round(iou, 4),
        "mean_latency_ms": round(np.mean(latencies), 2)
    }
    
    print("\nResults:")
    for k, v in results.items():
        print(f"  {k}: {v}")
        
    with open(OUTPUT_DIR / "culane_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Generate markdown table for report
    md = [
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| Pixel Precision | {precision:.4f} |",
        f"| Pixel Recall | {recall:.4f} |",
        f"| Pixel F1 Score | {f1:.4f} |",
        f"| Pixel IoU | {iou:.4f} |",
        f"| Mean Latency | {np.mean(latencies):.2f} ms |"
    ]
    (OUTPUT_DIR / "culane_results.md").write_text("\n".join(md))

if __name__ == "__main__":
    main()
