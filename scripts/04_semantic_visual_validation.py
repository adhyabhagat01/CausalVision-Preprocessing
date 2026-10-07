from pathlib import Path
from PIL import Image
import numpy as np
import csv
import random

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "CSDD_seg"
REPORTS_DIR = PROJECT_ROOT / "reports"

STAGE4_DIR = REPORTS_DIR / "stage4_visual_validation"
VIS_DIR = STAGE4_DIR / "visualizations"

IMAGE_DIR = DATASET_PATH / "img"
MASK_DIR = DATASET_PATH / "ground_truth"

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

# IMPORTANT:
# We are NOT assuming:
# 1 = Scratch
# 2 = Spot
# 3 = Rust
#
# Stage 4 is specifically for verifying this visually.

CLASS_VALUES = [1, 2, 3]

# Number of examples to visualize for each pixel value
SAMPLES_PER_CLASS = 10

# Fixed seed so the same examples can be reproduced
RANDOM_SEED = 42


# ============================================================
# CREATE DIRECTORIES
# ============================================================

STAGE4_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_image(split_dir, stem):
    """
    Find image corresponding to a mask stem.
    """

    for ext in IMAGE_EXTENSIONS:
        image_path = split_dir / f"{stem}{ext}"

        if image_path.exists():
            return image_path

    return None


def load_image(image_path):
    """
    Load image as RGB.
    """

    image = Image.open(image_path).convert("RGB")

    return np.array(image)


def load_mask(mask_path):
    """
    Load mask without changing its pixel values.
    """

    mask = Image.open(mask_path)

    return np.array(mask)


def create_overlay(image, mask):
    """
    Create a simple visualization overlay.

    The original image remains visible.
    Defect pixels are highlighted according to their
    original mask values.

    IMPORTANT:
    Colors here are only visualization colors.
    They do NOT represent Scratch/Spot/Rust semantics.
    """

    overlay = image.copy()

    # Visualization colors only
    visualization_colors = {
        1: [255, 0, 0],
        2: [0, 255, 0],
        3: [0, 0, 255]
    }

    for value in CLASS_VALUES:

        region = mask == value

        overlay[region] = (
            0.5 * overlay[region]
            + 0.5 * np.array(visualization_colors[value])
        ).astype(np.uint8)

    return overlay


def save_mask_image(mask, value, output_path):
    """
    Save only one particular mask value.

    White = selected pixel value
    Black = everything else
    """

    binary_mask = np.where(mask == value, 255, 0).astype(np.uint8)

    Image.fromarray(binary_mask).save(output_path)


def save_overlay(image, mask, output_path):
    """
    Save original image with mask overlay.
    """

    overlay = create_overlay(image, mask)

    Image.fromarray(overlay).save(output_path)


# ============================================================
# COLLECT ALL IMAGE-MASK PAIRS
# ============================================================

print("=" * 70)
print("STAGE 4 — SEMANTIC LABEL + VISUAL ANNOTATION VALIDATION")
print("=" * 70)

all_records = []

for split in SPLITS:

    image_split_dir = IMAGE_DIR / split
    mask_split_dir = MASK_DIR / split

    if not image_split_dir.exists():
        print(f"\nWARNING: Image folder not found: {image_split_dir}")
        continue

    if not mask_split_dir.exists():
        print(f"\nWARNING: Mask folder not found: {mask_split_dir}")
        continue

    print(f"\nProcessing split: {split}")

    mask_files = sorted(mask_split_dir.glob("*.png"))

    print(f"  Masks found: {len(mask_files)}")

    for mask_path in mask_files:

        # Remove "_mask" from mask filename
        mask_stem = mask_path.stem

        if mask_stem.endswith("_mask"):
            image_stem = mask_stem[:-5]
        else:
            image_stem = mask_stem

        image_path = find_image(image_split_dir, image_stem)

        if image_path is None:

            all_records.append({
                "split": split,
                "image": "",
                "mask": str(mask_path),
                "status": "MISSING_IMAGE",
                "value_1_pixels": 0,
                "value_2_pixels": 0,
                "value_3_pixels": 0
            })

            continue

        try:

            mask = load_mask(mask_path)

            values, counts = np.unique(mask, return_counts=True)

            pixel_counts = dict(zip(values, counts))

            value_1_count = int(pixel_counts.get(1, 0))
            value_2_count = int(pixel_counts.get(2, 0))
            value_3_count = int(pixel_counts.get(3, 0))

            all_records.append({
                "split": split,
                "image": str(image_path),
                "mask": str(mask_path),
                "status": "VALID",
                "value_1_pixels": value_1_count,
                "value_2_pixels": value_2_count,
                "value_3_pixels": value_3_count
            })

        except Exception as e:

            all_records.append({
                "split": split,
                "image": str(image_path),
                "mask": str(mask_path),
                "status": f"MASK_READ_ERROR: {e}",
                "value_1_pixels": 0,
                "value_2_pixels": 0,
                "value_3_pixels": 0
            })


# ============================================================
# SAVE CSV REPORT
# ============================================================

csv_path = STAGE4_DIR / "stage4_visual_validation_report.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "split",
            "image",
            "mask",
            "status",
            "value_1_pixels",
            "value_2_pixels",
            "value_3_pixels"
        ]
    )

    writer.writeheader()

    writer.writerows(all_records)


# ============================================================
# FIND EXAMPLES FOR EACH MASK VALUE
# ============================================================

random.seed(RANDOM_SEED)

examples = {
    1: [],
    2: [],
    3: []
}

for record in all_records:

    if record["status"] != "VALID":
        continue

    for value in CLASS_VALUES:

        pixel_count = record[f"value_{value}_pixels"]

        if pixel_count > 0:
            examples[value].append(record)


# ============================================================
# GENERATE VISUALIZATIONS
# ============================================================

print("\nGenerating visual examples...")

for value in CLASS_VALUES:

    value_dir = VIS_DIR / f"pixel_value_{value}"
    value_dir.mkdir(parents=True, exist_ok=True)

    candidates = examples[value].copy()

    random.shuffle(candidates)

    selected = candidates[:SAMPLES_PER_CLASS]

    print(
        f"  Pixel value {value}: "
        f"{len(candidates)} images available, "
        f"{len(selected)} visual examples selected"
    )

    for index, record in enumerate(selected, start=1):

        image_path = Path(record["image"])
        mask_path = Path(record["mask"])

        try:

            image = load_image(image_path)
            mask = load_mask(mask_path)

            # ------------------------------------------------
            # Save original image
            # ------------------------------------------------

            original_output = (
                value_dir /
                f"{index:02d}_{image_path.stem}_original.jpg"
            )

            Image.fromarray(image).save(original_output)

            # ------------------------------------------------
            # Save raw mask
            # ------------------------------------------------

            raw_mask_output = (
                value_dir /
                f"{index:02d}_{image_path.stem}_raw_mask.png"
            )

            Image.fromarray(mask.astype(np.uint8)).save(raw_mask_output)

            # ------------------------------------------------
            # Save selected class mask
            # ------------------------------------------------

            class_mask_output = (
                value_dir /
                f"{index:02d}_{image_path.stem}_value_{value}.png"
            )

            save_mask_image(
                mask,
                value,
                class_mask_output
            )

            # ------------------------------------------------
            # Save overlay
            # ------------------------------------------------

            overlay_output = (
                value_dir /
                f"{index:02d}_{image_path.stem}_overlay.jpg"
            )

            save_overlay(
                image,
                mask,
                overlay_output
            )

        except Exception as e:

            print(
                f"    ERROR processing {image_path.name}: {e}"
            )


# ============================================================
# GENERATE SUMMARY
# ============================================================

summary_path = STAGE4_DIR / "stage4_summary.txt"

total_valid = sum(
    1 for record in all_records
    if record["status"] == "VALID"
)

total_missing = sum(
    1 for record in all_records
    if record["status"] == "MISSING_IMAGE"
)

total_errors = sum(
    1 for record in all_records
    if "ERROR" in record["status"]
)

with open(summary_path, "w", encoding="utf-8") as file:

    file.write("STAGE 4 — SEMANTIC LABEL + VISUAL ANNOTATION VALIDATION\n")
    file.write("=" * 65 + "\n\n")

    file.write("IMPORTANT:\n")
    file.write(
        "Pixel values 1, 2 and 3 have NOT been assigned to "
        "Scratch, Spot or Rust automatically.\n"
    )

    file.write(
        "The mapping must be confirmed through visual inspection "
        "or official dataset documentation.\n\n"
    )

    file.write(f"Total records checked: {len(all_records)}\n")
    file.write(f"Valid image-mask pairs: {total_valid}\n")
    file.write(f"Missing images: {total_missing}\n")
    file.write(f"Read/processing errors: {total_errors}\n\n")

    file.write("Visual examples generated:\n")

    for value in CLASS_VALUES:

        file.write(
            f"Pixel value {value}: "
            f"{len(examples[value][:SAMPLES_PER_CLASS])} examples\n"
        )

    file.write("\n")
    file.write("SEMANTIC MAPPING TO VERIFY:\n")
    file.write("----------------------------\n")
    file.write("Pixel value 1 -> __________________\n")
    file.write("Pixel value 2 -> __________________\n")
    file.write("Pixel value 3 -> __________________\n")

    file.write("\nPossible classes:\n")
    file.write("Scratch\n")
    file.write("Spot\n")
    file.write("Rust\n")


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("STAGE 4 COMPLETED")
print("=" * 70)

print(f"\nCSV report:")
print(csv_path)

print(f"\nSummary:")
print(summary_path)

print(f"\nVisualizations:")
print(VIS_DIR)

print("\nIMPORTANT:")
print("Open the visualizations and determine:")
print("  Pixel value 1 = ?")
print("  Pixel value 2 = ?")
print("  Pixel value 3 = ?")

print("\nDo NOT assign the classes based only on pixel frequency.")
print("Use the actual visual appearance of the defects.")

print("\nOriginal dataset was NOT modified.")