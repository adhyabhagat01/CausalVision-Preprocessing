from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import csv
import json
import sys


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "CSDD_seg"

IMAGE_ROOT = DATASET_PATH / "img"
MASK_ROOT = DATASET_PATH / "ground_truth"

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXPECTED SPLITS
# ============================================================

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}

MASK_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
}


# ============================================================
# CHECK ROOT DIRECTORIES
# ============================================================

if not IMAGE_ROOT.exists():
    print(f"ERROR: Image directory not found: {IMAGE_ROOT}")
    sys.exit(1)

if not MASK_ROOT.exists():
    print(f"ERROR: Ground-truth directory not found: {MASK_ROOT}")
    sys.exit(1)


print("=" * 75)
print("STAGE 2 - IMAGE / GROUND-TRUTH VALIDATION")
print("=" * 75)

print(f"\nImage root : {IMAGE_ROOT}")
print(f"Mask root  : {MASK_ROOT}")


# ============================================================
# GLOBAL RESULTS
# ============================================================

all_results = []

total_images = 0
total_masks = 0

total_pairs = 0
total_missing_masks = 0
total_missing_images = 0

total_dimension_mismatch = 0
total_unreadable_masks = 0

global_mask_values = Counter()
global_mask_modes = Counter()

split_summary = {}


# ============================================================
# PROCESS EACH SPLIT
# ============================================================

for split in SPLITS:

    print("\n" + "=" * 75)
    print(f"PROCESSING SPLIT: {split.upper()}")
    print("=" * 75)

    image_dir = IMAGE_ROOT / split
    mask_dir = MASK_ROOT / split

    if not image_dir.exists():
        print(f"WARNING: Image split does not exist: {image_dir}")
        continue

    if not mask_dir.exists():
        print(f"WARNING: Ground-truth split does not exist: {mask_dir}")
        continue


    # --------------------------------------------------------
    # FIND IMAGE FILES
    # --------------------------------------------------------

    image_files = sorted(
        [
            p for p in image_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


    # --------------------------------------------------------
    # FIND MASK FILES
    # --------------------------------------------------------

    mask_files = sorted(
        [
            p for p in mask_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in MASK_EXTENSIONS
        ]
    )


    print(f"\nImages found       : {len(image_files)}")
    print(f"Ground-truth masks : {len(mask_files)}")


    total_images += len(image_files)
    total_masks += len(mask_files)


    # --------------------------------------------------------
    # CREATE STEM MAPS
    # --------------------------------------------------------

    image_map = {}

    for path in image_files:
        stem = path.stem.lower()

        if stem in image_map:
            print(f"WARNING: Duplicate image stem: {stem}")

        image_map[stem] = path


    mask_map = {}

    for path in mask_files:
        stem = path.stem.lower()

        if stem.endswith("_mask"):
            stem = stem[:-5]

        if stem in mask_map:

            print(f"WARNING: Duplicate mask stem: {stem}")

        mask_map[stem] = path


    image_stems = set(image_map.keys())
    mask_stems = set(mask_map.keys())


    # --------------------------------------------------------
    # FIND MISSING FILES
    # --------------------------------------------------------

    missing_masks = sorted(image_stems - mask_stems)
    missing_images = sorted(mask_stems - image_stems)


    print(f"\nImages without masks : {len(missing_masks)}")
    print(f"Masks without images : {len(missing_images)}")


    if missing_masks:
        print("\nFirst missing masks:")

        for name in missing_masks[:10]:
            print(f"  {name}")


    if missing_images:
        print("\nFirst missing images:")

        for name in missing_images[:10]:
            print(f"  {name}")


    total_missing_masks += len(missing_masks)
    total_missing_images += len(missing_images)


    # --------------------------------------------------------
    # VALID PAIRS
    # --------------------------------------------------------

    paired_stems = sorted(image_stems & mask_stems)

    print(f"\nValid image-mask pairs : {len(paired_stems)}")

    total_pairs += len(paired_stems)


    # --------------------------------------------------------
    # MASK ANALYSIS
    # --------------------------------------------------------

    split_mask_values = Counter()
    split_mask_modes = Counter()

    split_dimension_mismatches = []
    split_unreadable_masks = []
    split_empty_masks = []

    mask_pixel_counts = defaultdict(int)


    for index, stem in enumerate(paired_stems, start=1):

        image_path = image_map[stem]
        mask_path = mask_map[stem]


        # ----------------------------------------------------
        # READ IMAGE DIMENSIONS
        # ----------------------------------------------------

        try:

            with Image.open(image_path) as image:

                image_width, image_height = image.size
                image_mode = image.mode

        except Exception as error:

            all_results.append({
                "split": split,
                "stem": stem,
                "image": str(image_path),
                "mask": str(mask_path),
                "status": "IMAGE_READ_ERROR",
                "error": str(error)
            })

            continue


        # ----------------------------------------------------
        # READ MASK
        # ----------------------------------------------------

        try:

            with Image.open(mask_path) as mask:

                mask_width, mask_height = mask.size
                mask_mode = mask.mode

                # Convert to a simple pixel-value representation.
                #
                # For P / L / 1 masks this gives class/index values.
                # For RGB/RGBA masks we record that separately.
                if mask_mode in ["1", "L", "P", "I", "I;16"]:

                    mask_values = Counter(mask.getdata())

                    for value, count in mask_values.items():

                        split_mask_values[value] += count
                        global_mask_values[value] += count

                    for value, count in mask_values.items():
                        mask_pixel_counts[value] += count

                else:

                    # RGB/RGBA or other multi-channel masks
                    mask_values = Counter(mask.getdata())

                    split_mask_values["MULTI_CHANNEL"] += len(
                        mask_values
                    )

                split_mask_modes[mask_mode] += 1
                global_mask_modes[mask_mode] += 1


        except Exception as error:

            split_unreadable_masks.append({
                "stem": stem,
                "mask": str(mask_path),
                "error": str(error)
            })

            total_unreadable_masks += 1

            all_results.append({
                "split": split,
                "stem": stem,
                "image": str(image_path),
                "mask": str(mask_path),
                "status": "MASK_READ_ERROR",
                "error": str(error)
            })

            continue


        # ----------------------------------------------------
        # DIMENSION CHECK
        # ----------------------------------------------------

        dimensions_match = (
            image_width == mask_width
            and image_height == mask_height
        )


        if not dimensions_match:

            split_dimension_mismatches.append({
                "stem": stem,
                "image_size": f"{image_width}x{image_height}",
                "mask_size": f"{mask_width}x{mask_height}"
            })

            total_dimension_mismatch += 1


        # ----------------------------------------------------
        # EMPTY MASK CHECK
        # ----------------------------------------------------

        is_empty_mask = False

        if mask_mode in ["1", "L", "P", "I", "I;16"]:

            unique_values = set(mask_values.keys())

            # A mask containing only zero is treated as
            # potentially empty.
            if unique_values == {0}:

                is_empty_mask = True
                split_empty_masks.append(stem)


        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        all_results.append({
            "split": split,
            "stem": stem,
            "image": str(image_path),
            "mask": str(mask_path),
            "image_size": f"{image_width}x{image_height}",
            "mask_size": f"{mask_width}x{mask_height}",
            "image_mode": image_mode,
            "mask_mode": mask_mode,
            "dimensions_match": dimensions_match,
            "empty_mask": is_empty_mask,
            "status": "OK"
                if dimensions_match
                else "DIMENSION_MISMATCH"
        })


        # Progress
        if index % 250 == 0:

            print(
                f"Processed {index}/{len(paired_stems)} pairs..."
            )


    # --------------------------------------------------------
    # SPLIT SUMMARY
    # --------------------------------------------------------

    split_summary[split] = {
        "images": len(image_files),
        "masks": len(mask_files),
        "valid_pairs": len(paired_stems),
        "missing_masks": len(missing_masks),
        "missing_images": len(missing_images),
        "dimension_mismatches": len(
            split_dimension_mismatches
        ),
        "unreadable_masks": len(
            split_unreadable_masks
        ),
        "empty_masks": len(
            split_empty_masks
        ),
        "mask_modes": dict(split_mask_modes),
        "mask_values": {
            str(k): v
            for k, v in split_mask_values.items()
        }
    }


    # --------------------------------------------------------
    # PRINT SPLIT RESULTS
    # --------------------------------------------------------

    print("\n--- Split Result ---")

    print(
        f"Valid pairs          : {len(paired_stems)}"
    )

    print(
        f"Missing masks        : {len(missing_masks)}"
    )

    print(
        f"Missing images       : {len(missing_images)}"
    )

    print(
        f"Dimension mismatches : "
        f"{len(split_dimension_mismatches)}"
    )

    print(
        f"Unreadable masks     : "
        f"{len(split_unreadable_masks)}"
    )

    print(
        f"Empty masks          : "
        f"{len(split_empty_masks)}"
    )

    print(
        f"Mask modes           : "
        f"{dict(split_mask_modes)}"
    )

    print(
        f"Mask values          : "
        f"{dict(split_mask_values)}"
    )


# ============================================================
# GLOBAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("GLOBAL VALIDATION SUMMARY")
print("=" * 75)

print(f"\nTotal images              : {total_images}")
print(f"Total ground-truth masks  : {total_masks}")
print(f"Total valid pairs         : {total_pairs}")
print(f"Missing masks             : {total_missing_masks}")
print(f"Missing images            : {total_missing_images}")
print(
    f"Dimension mismatches      : "
    f"{total_dimension_mismatch}"
)
print(
    f"Unreadable masks          : "
    f"{total_unreadable_masks}"
)

print("\nGlobal mask modes:")

for mode, count in global_mask_modes.items():
    print(f"  {mode:<10} : {count}")


# ============================================================
# GLOBAL MASK VALUES
# ============================================================

print("\nGlobal mask pixel values:")

for value, count in global_mask_values.most_common():

    print(
        f"  {str(value):<10} : "
        f"{count:,} pixels"
    )


# ============================================================
# SAVE PAIR VALIDATION CSV
# ============================================================

pair_csv = REPORTS_DIR / "stage2_image_mask_pairs.csv"

with open(
    pair_csv,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "split",
            "stem",
            "image",
            "mask",
            "image_size",
            "mask_size",
            "image_mode",
            "mask_mode",
            "dimensions_match",
            "empty_mask",
            "status"
        ]
    )

    writer.writeheader()

    for row in all_results:

        # Ensure all fields exist
        writer.writerow({
            field: row.get(field, "")
            for field in writer.fieldnames
        })


# ============================================================
# SAVE SUMMARY JSON
# ============================================================

summary = {
    "dataset_path": str(DATASET_PATH),

    "total_images": total_images,

    "total_masks": total_masks,

    "total_valid_pairs": total_pairs,

    "missing_masks": total_missing_masks,

    "missing_images": total_missing_images,

    "dimension_mismatches": total_dimension_mismatch,

    "unreadable_masks": total_unreadable_masks,

    "global_mask_modes": dict(global_mask_modes),

    "global_mask_values": {
        str(value): count
        for value, count in global_mask_values.items()
    },

    "splits": split_summary
}


summary_json = (
    REPORTS_DIR
    / "stage2_pair_and_mask_summary.json"
)


with open(
    summary_json,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 75)
print("STAGE 2 COMPLETED")
print("=" * 75)

print("\nGenerated:")

print(f"1. {pair_csv}")
print(f"2. {summary_json}")

print("\nNo original dataset files were modified.")
print("\nNext step:")
print(
    "Use the actual mask pixel values to determine "
    "Scratch / Spot / Rust encoding before converting "
    "the dataset to YOLO segmentation format."
)

print("=" * 75)