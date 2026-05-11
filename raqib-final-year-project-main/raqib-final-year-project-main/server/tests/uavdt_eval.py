"""
Formal Evaluation on UAVDT (Ground-Truth Dataset)
================================================
Replaces the "Silver Reference" logic with absolute accuracy metrics
measured against the UAVDT benchmark dataset.

Metrics:
  • Precision: TP / (TP + FP)
  • Recall:    TP / (TP + FN)
  • F1 Score:  2 * (P * R) / (P + R)
  • mAP@50:    Mean Average Precision at IoU = 0.5

UAVDT Class Mapping:
  • 'car'     -> 'car'
  • 'truck'   -> 'truck'
  • 'bus'     -> 'bus'
  • 'vehicle' -> 'car' (generic fallback)
"""
import os
import sys
import json
import time
import logging
from pathlib import Path
import cv2
import numpy as np
import torch

# Add server to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detectors.registry import DetectorRegistry

# Configuration
DATASET_ROOT = Path(r"L:\Misc\uavdt-DatasetNinja")
TEST_IMG_DIR = DATASET_ROOT / "test" / "img"
TEST_ANN_DIR = DATASET_ROOT / "test" / "ann"
OUTPUT_DIR = ROOT / "tests" / "_inspect" / "uavdt_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sampling: Pick representative sequences and frames
# Sequences: M0203 (night), S1607 (day), S1701 (day), S1702 (day)
SAMPLES = [
    ("M0203", 50), # 50 frames from night sequence
    ("S1607", 50), # 50 frames from day sequence
]

CLASS_MAP = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "vehicle": "car"
}

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)

def load_ground_truth(seq_name, frame_idx):
    filename = f"{seq_name}_img{frame_idx:06d}.jpg.json"
    path = TEST_ANN_DIR / filename
    if not path.exists():
        return None
    
    with open(path, "r") as f:
        data = json.load(f)
    
    objs = []
    for obj in data.get("objects", []):
        label = CLASS_MAP.get(obj["classTitle"])
        if not label: continue
        
        # Format: [x1, y1], [x2, y2]
        pts = obj["points"]["exterior"]
        box = [pts[0][0], pts[0][1], pts[1][0], pts[1][1]]
        objs.append({"label": label, "box": box})
    return objs

def evaluate_model(name, detector, sampled_frames, iou_threshold=0.5):
    print(f"Evaluating {name}...")
    tp = fp = fn = 0
    latencies = []
    
    for seq_name, frame_idx, img_path, gt in sampled_frames:
        frame = cv2.imread(str(img_path))
        if frame is None: continue
        
        t0 = time.perf_counter()
        preds = detector.detect(frame)
        latencies.append((time.perf_counter() - t0) * 1000)
        
        # Filter preds to standard vehicle labels
        preds = [p for p in preds if p["label"] in ("car", "truck", "bus")]
        
        matched_gt = set()
        for p in preds:
            best_iou = 0
            best_gt_idx = -1
            for idx, g in enumerate(gt):
                if idx in matched_gt: continue
                if p["label"] != g["label"]: continue
                
                curr_iou = iou(p["box"], g["box"])
                if curr_iou > best_iou:
                    best_iou = curr_iou
                    best_gt_idx = idx
            
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        
        fn += (len(gt) - len(matched_gt))
        
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "model": name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_latency_ms": np.mean(latencies),
        "tp": tp, "fp": fp, "fn": fn
    }

def main():
    registry = DetectorRegistry()
    models = {
        "yolo11": registry.get_vehicle("yolo11"),
        "visdrone": registry.get_vehicle("visdrone"),
        "fasterrcnn": registry.get_vehicle("fasterrcnn")
    }
    
    print("Harvesting frames...")
    sampled_frames = []
    for seq, count in SAMPLES:
        # Step through sequence
        for i in range(1, count * 5, 5): # sample every 5th frame to cover 250 frames of temporal span
            img_name = f"{seq}_img{i:06d}.jpg"
            img_path = TEST_IMG_DIR / img_name
            gt = load_ground_truth(seq, i)
            if gt and img_path.exists():
                sampled_frames.append((seq, i, img_path, gt))
    
    print(f"Total sampled frames: {len(sampled_frames)}")
    
    results = []
    for name, det in models.items():
        res = evaluate_model(name, det, sampled_frames)
        results.append(res)
        
    # Write report table
    md = [
        "| Model | Precision | Recall | F1 Score | Mean Latency (ms) |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ]
    for r in results:
        md.append(f"| {r['model']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['mean_latency_ms']:.1f} |")
    
    (OUTPUT_DIR / "uavdt_results.md").write_text("\n".join(md))
    with open(OUTPUT_DIR / "uavdt_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nEvaluation complete. Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
