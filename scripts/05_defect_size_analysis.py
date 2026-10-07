import csv
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "CSDD_seg"
REPORTS_DIR = PROJECT_ROOT / "reports" / "stage5_defect_size_analysis"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MASK_DIR = DATASET_PATH / "ground_truth"
SPLITS = ["train", "val", "test"]
CLASS_VALUES = [1, 2, 3]

results = []

for split in SPLITS:
    split_dir = MASK_DIR / split
    if not split_dir.exists():
        continue
    
    for mask_path in sorted(split_dir.glob("*.png")):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        for val in CLASS_VALUES:
            binary = (mask == val).astype(np.uint8)
            if not np.any(binary):
                continue
            
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
            
            for i in range(1, num_labels):
                x, y, w, h, area = stats[i]
                results.append({
                    "split": split,
                    "image": mask_path.stem.replace("_mask", ""),
                    "class_val": val,
                    "x": x, "y": y, "width": w, "height": h,
                    "area_pixels": area
                })

# Save output CSV
csv_path = REPORTS_DIR / "defect_bbox_stats.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["split", "image", "class_val", "x", "y", "width", "height", "area_pixels"])
    writer.writeheader()
    writer.writerows(results)

print(f"Stage 5 completed. Report saved to: {csv_path}")