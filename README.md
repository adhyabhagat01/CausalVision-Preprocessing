# CausalVision - CSDD Dataset Preprocessing

## Dataset Preprocessing Log

This README records the preprocessing, cleaning, validation, and dataset preparation steps performed on the CSDD dataset.

The original CSDD dataset is preserved under:

```text
CSDD_seg/
├── img/
│   ├── train/
│   ├── val/
│   └── test/
│
├── ground_truth/
│   ├── train/
│   ├── val/
│   └── test/
│
└── readme.txt.txt



Stage 1 - Dataset Inventory
Objective
The first stage performs a complete inventory of the CSDD dataset before modifying or preprocessing any data.
The following were checked:
- Total number of files
- Image file types
- Ground-truth file types
- Image dimensions
- Image modes
- Existing train/validation/test organization
- Image integrity
- Dataset size
- Folder structure
Stage 1 Results
Dataset Location
C:\Users\Adhya\OneDrive\Desktop\Projects\Gen_AI\project\CSDD_seg

Overall Dataset
Total files: 4201
Total dataset size: 2.555 GB

File Distribution
JPG images: 2100
PNG files: 2100
Other files: 1

The 2100 PNG files are located inside the ground_truth directory and represent the ground-truth segmentation masks. Therefore, the dataset contains:
2100 original casting images
2100 corresponding ground-truth segmentation masks

Image Dimensions
All 4200 image/mask files detected by the inventory script have:
3648 × 3648 pixels

Image Modes
RGB: 2100
P:   2100

The RGB images correspond to the original casting images, while the P-mode PNG files are the ground-truth mask images.
Existing Dataset Split
The dataset already contains predefined train/validation/test splits.
Split       Images       Ground Truth
-------------------------------------
Train       1343         1343
Validation   335          335
Test         422          422
-------------------------------------
Total        2100         2100

Therefore, the existing CSDD split will be preserved rather than creating a new random split at this stage.
Dataset Integrity
Corrupted images detected: 0

No corrupted/unreadable images were detected during the initial image integrity check.
Folder Structure
CSDD_seg/
│
├── img/
│   ├── train/       1343
│   ├── val/          335
│   └── test/         422
│
├── ground_truth/
│   ├── train/       1343
│   ├── val/          335
│   └── test/         422
│
└── readme.txt.txt

Stage 1 Conclusion
The initial inventory confirms that the CSDD dataset contains 2100 casting images with corresponding ground-truth segmentation masks and predefined train, validation, and test splits.
No corrupted images were detected.
The original dataset will remain unchanged. Subsequent preprocessing will operate through generated/processed copies and conversion outputs.
Generated Stage 1 Reports
The following reports were generated automatically:
reports/
├── stage1_file_inventory.csv
├── stage1_folder_summary.csv
├── stage1_corrupted_images.csv
└── stage1_dataset_summary.json

These reports preserve the machine-readable results of Stage 1.
Preprocessing Stages
The preprocessing pipeline will be performed step by step:
1. Dataset Inventory
2. Image–Ground Truth Pair Validation
3. Ground-Truth Mask Analysis
4. Class/Defect Label Validation
5. Image and Annotation Quality Checks
6. Duplicate Analysis
7. Defect/Class Distribution Analysis
8. Visual Annotation Verification
9. Defect Size Analysis
10. Dataset Normalization
11. YOLO Segmentation Format Conversion
12. Final Dataset Validation
13. Dataset Statistics and Report Generation
14. YOLO Baseline Training
15. Evaluation on the Original Real Test Set

STAGE 2:
C:\Users\Adhya\OneDrive\Desktop\Projects\Gen_AI\project\scripts\02_validate_pairs_and_masks.py:286: DeprecationWarning: Image.Image.getdata is deprecated and will be removed in Pillow 14 (2027-10-15). Use get_flattened_data instead.
  mask_values = Counter(mask.getdata())
Processed 250/1343 pairs...
Processed 500/1343 pairs...
Processed 750/1343 pairs...
Processed 1000/1343 pairs...
Processed 1250/1343 pairs...

--- Split Result ---
Valid pairs          : 1343
Missing masks        : 0
Missing images       : 0
Dimension mismatches : 0
Unreadable masks     : 0
Empty masks          : 0
Mask modes           : {'P': 1343}
Mask values          : {0: 17833766522, 1: 19708500, 2: 2690313, 3: 16349737}

===========================================================================
PROCESSING SPLIT: VAL
===========================================================================

Images found       : 335
Ground-truth masks : 335

Images without masks : 0
Masks without images : 0

Valid image-mask pairs : 335
Processed 250/335 pairs...

--- Split Result ---
Valid pairs          : 335
Missing masks        : 0
Missing images       : 0
Dimension mismatches : 0
Unreadable masks     : 0
Empty masks          : 0
Mask modes           : {'P': 335}
Mask values          : {0: 4448646155, 1: 4866016, 2: 663363, 3: 3972306}

===========================================================================
PROCESSING SPLIT: TEST
===========================================================================

Images found       : 422
Ground-truth masks : 422

Images without masks : 0
Masks without images : 0

Valid image-mask pairs : 422
Processed 250/422 pairs...

--- Split Result ---
Valid pairs          : 422
Missing masks        : 0
Missing images       : 0
Dimension mismatches : 0
Unreadable masks     : 0
Empty masks          : 0
Mask modes           : {'P': 422}
Mask values          : {0: 5603367882, 1: 6465774, 2: 858683, 3: 5243149}

===========================================================================
GLOBAL VALIDATION SUMMARY
===========================================================================

Total images              : 2100
Total ground-truth masks  : 2100
Total valid pairs         : 2100
Missing masks             : 0
Missing images            : 0
Dimension mismatches      : 0
Unreadable masks          : 0

Global mask modes:
  P          : 2100

Global mask pixel values:
  0          : 27,885,780,559 pixels
  1          : 31,040,290 pixels
  3          : 25,565,192 pixels
  2          : 4,212,359 pixels

===========================================================================
STAGE 2 COMPLETED
===========================================================================

Generated:
1. C:\Users\Adhya\OneDrive\Desktop\Projects\Gen_AI\project\reports\stage2_image_mask_pairs.csv
2. C:\Users\Adhya\OneDrive\Desktop\Projects\Gen_AI\project\reports\stage2_pair_and_mask_summary.json

No original dataset files were modified.

Next step:
Use the actual mask pixel values to determine Scratch / Spot / Rust encoding before converting the dataset to YOLO segmentation format.
===========================================================================

STAGE 3:
PS C:\Users\Adhya\OneDrive\Desktop\Projects\Gen_AI\project> ^C    
PS C:\Users\Adhya\OneDrive\Desktop\Projects\Gen_AI\project> python scripts\03_class_mapping_and_distribution.py
======================================================================
STAGE 3 - CLASS MAPPING AND CLASS DISTRIBUTION
======================================================================
Dataset: C:\Users\Adhya\OneDrive\Desktop\Projects\Gen_AI\project\CSDD_seg

No local documentation explicitly confirms the 1/2/3 class mapping.
We will NOT guess the mapping.

======================================================================
PROCESSING: TRAIN
======================================================================
Processed 250/1343 masks...
Processed 500/1343 masks...
Processed 750/1343 masks...
Processed 1000/1343 masks...
Processed 1250/1343 masks...
Processed 1343/1343 masks...

--- Split Result ---
Total pixels      : 17,872,515,072
Pixel value 0     : 17,833,766,522
Pixel value 1     : 19,708,500
Pixel value 2     : 2,690,313
Pixel value 3     : 16,349,737
Images containing 1/2/3: 1342 / 1244 / 731

======================================================================
PROCESSING: VAL
======================================================================
Processed 250/335 masks...
Processed 335/335 masks...

--- Split Result ---
Total pixels      : 4,458,147,840
Pixel value 0     : 4,448,646,155
Pixel value 1     : 4,866,016
Pixel value 2     : 663,363
Pixel value 3     : 3,972,306
Images containing 1/2/3: 333 / 312 / 184

======================================================================
PROCESSING: TEST
======================================================================
Processed 250/422 masks...
Processed 422/422 masks...

--- Split Result ---
Total pixels      : 5,615,935,488
Pixel value 0     : 5,603,367,882
Pixel value 1     : 6,465,774
Pixel value 2     : 858,683
Pixel value 3     : 5,243,149
Images containing 1/2/3: 422 / 393 / 233

======================================================================
GLOBAL RESULT
======================================================================
Pixel value 0: 27,885,780,559 pixels (99.782378%)
Pixel value 1: 31,040,290 pixels (0.111070%)
Pixel value 2: 4,212,359 pixels (0.015073%)
Pixel value 3: 25,565,192 pixels (0.091479%)
Defect pixels (1+2+3): 60,817,841
Defect pixel percentage: 0.217622%

Images containing each value:
  1: 2097
  2: 1949
  3: 1148

Palette RGB:
  0: [0, 0, 0]
  1: [1, 1, 1]
  2: [2, 2, 2]
  3: [3, 3, 3]

IMPORTANT: Do not assume 1=Scratch, 2=Spot, 3=Rust until mapping is verified.

Generated:
1. C:\Users\Adhya\OneDrive\Desktop\Projects\Gen_AI\project\reports\stage3_class_distribution_summary.json
2. C:\Users\Adhya\OneDrive\Desktop\Projects\Gen_AI\project\reports\stage3_per_image_class_pixels.csv
Original dataset files were not modified.
======================================================================








