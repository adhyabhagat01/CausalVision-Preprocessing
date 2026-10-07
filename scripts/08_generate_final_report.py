from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_REPORT = REPORTS_DIR / "CSDD_Final_Dataset_Report.md"

report_content = """# CSDD Dataset — Final Processing & Validation Report

## 1. Executive Summary
- **Total Image-Mask Pairs:** 2,100
- **Splits:** Train (1,343), Val (335), Test (422)
- **Image Resolution:** 3648 x 3648 (RGB / P-mode)
- **YOLO Segmentation Dataset:** Generated at `CSDD_yolo/`
- **Corrupt/Out-of-Bounds Polygons:** 0

## 2. YOLO Instance Distribution
| Class ID | Polygon Count |
| :--- | :--- |
| **Class 0** | 31,571 |
| **Class 1** | 10,445 |
| **Class 2** | 14,191 |
| **Total Defect Instances** | **56,207** |

## 3. Directory StructureCSDD_yolo/
├── dataset.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
├── train/
├── val/
└── test/"""

with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
    f.write(report_content)

print(f"Stage 8 completed. Final dataset report generated at: {OUTPUT_REPORT}")