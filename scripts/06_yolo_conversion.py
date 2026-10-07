import shutil
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "CSDD_seg"
YOLO_DIR = PROJECT_ROOT / "CSDD_yolo"

# Map mask pixel values (1, 2, 3) to 0-indexed YOLO class IDs (0, 1, 2)
CLASS_MAP = {1: 0, 2: 1, 3: 2}
SPLITS = ["train", "val", "test"]

for split in SPLITS:
    (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

for split in SPLITS:
    img_dir = DATASET_PATH / "img" / split
    mask_dir = DATASET_PATH / "ground_truth" / split

    if not mask_dir.exists():
        continue

    for mask_path in sorted(mask_dir.glob("*.png")):
        mask_stem = mask_path.stem
        img_stem = mask_stem[:-5] if mask_stem.endswith("_mask") else mask_stem
        
        img_path = None
        for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
            cand = img_dir / f"{img_stem}{ext}"
            if cand.exists():
                img_path = cand
                break
        
        if img_path is None:
            continue

        # Copy image to YOLO directory
        shutil.copy(img_path, YOLO_DIR / "images" / split / img_path.name)

        # Read mask and extract normalized polygons
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        h, w = mask.shape
        label_lines = []

        for val, class_id in CLASS_MAP.items():
            binary = (mask == val).astype(np.uint8)
            if not np.any(binary):
                continue

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if len(cnt) < 3:
                    continue
                cnt = cnt.squeeze(axis=1)
                poly = []
                for pt in cnt:
                    poly.extend([round(float(pt[0]) / w, 6), round(float(pt[1]) / h, 6)])
                label_lines.append(f"{class_id} " + " ".join(map(str, poly)))

        # Write YOLO label file
        label_path = YOLO_DIR / "labels" / split / f"{img_stem}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines) + ("\n" if label_lines else ""))

# Write dataset.yaml configuration
yaml_content = f"""path: {YOLO_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: class_1
  1: class_2
  2: class_3
"""

with open(YOLO_DIR / "dataset.yaml", "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"Stage 6 completed. YOLO dataset generated at: {YOLO_DIR}")