from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLO_DIR = PROJECT_ROOT / "CSDD_yolo"
REPORTS_DIR = PROJECT_ROOT / "reports" / "stage7_yolo_validation"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]
class_counts = {0: 0, 1: 0, 2: 0}
total_images = 0
total_labels = 0
corrupt_files = 0

for split in SPLITS:
    img_dir = YOLO_DIR / "images" / split
    label_dir = YOLO_DIR / "labels" / split

    imgs = list(img_dir.glob("*.*"))
    labels = list(label_dir.glob("*.txt"))

    total_images += len(imgs)
    total_labels += len(labels)

    for lbl_path in labels:
        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            
            cid = int(parts[0])
            coords = [float(x) for x in parts[1:]]

            # Verify class ID and normalized polygon range [0, 1]
            if cid in class_counts:
                class_counts[cid] += 1
            if any(c < 0.0 or c > 1.0 for c in coords):
                corrupt_files += 1

summary_text = f"""STAGE 7 — YOLO DATASET VALIDATION REPORT
==========================================
Total Images: {total_images}
Total Labels: {total_labels}
Corrupt/Out-of-Bounds Files: {corrupt_files}

Instance Counts per YOLO Class ID:
  Class 0: {class_counts[0]}
  Class 1: {class_counts[1]}
  Class 2: {class_counts[2]}
"""

summary_path = REPORTS_DIR / "stage7_yolo_validation_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary_text)

print(summary_text)
print(f"Stage 7 completed. Report saved to: {summary_path}")