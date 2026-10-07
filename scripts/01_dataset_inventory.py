from pathlib import Path
from collections import Counter
from PIL import Image
import csv
import json
import sys


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "CSDD_seg"

REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SUPPORTED FILE TYPES
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}

ANNOTATION_EXTENSIONS = {
    ".txt",
    ".json",
    ".xml",
    ".csv",
    ".yaml",
    ".yml",
    ".mask"
}

SPLIT_NAMES = {
    "train",
    "training",
    "val",
    "valid",
    "validation",
    "test",
    "testing"
}


# ============================================================
# CHECK DATASET PATH
# ============================================================

if not DATASET_PATH.exists():
    print("\nERROR: Dataset path does not exist.")
    print(f"Path entered: {DATASET_PATH}")
    print("\nPlease change DATASET_PATH at the top of this file.")
    sys.exit(1)

if not DATASET_PATH.is_dir():
    print("\nERROR: DATASET_PATH is not a folder.")
    sys.exit(1)


print("=" * 70)
print("CSDD DATASET INVENTORY")
print("=" * 70)

print(f"\nDataset path:")
print(DATASET_PATH)


# ============================================================
# DATA STRUCTURES
# ============================================================

all_files = []

image_count = 0
annotation_count = 0
other_count = 0

image_extensions = Counter()
annotation_extensions = Counter()
other_extensions = Counter()

folder_counts = Counter()
split_counts = Counter()

image_dimensions = Counter()
image_modes = Counter()

corrupted_images = []
unreadable_images = []

annotation_samples = []
image_samples = []

total_image_size = 0
total_annotation_size = 0
total_other_size = 0


# ============================================================
# RECURSIVE DATASET SCAN
# ============================================================

print("\nScanning dataset...")
print("This may take some time for a large dataset.\n")

for file_path in DATASET_PATH.rglob("*"):

    if not file_path.is_file():
        continue

    all_files.append(file_path)

    extension = file_path.suffix.lower()

    try:
        file_size = file_path.stat().st_size
    except OSError:
        file_size = 0

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    if extension in IMAGE_EXTENSIONS:

        image_count += 1
        image_extensions[extension] += 1
        total_image_size += file_size

        if len(image_samples) < 20:
            image_samples.append(str(file_path))

        # Determine folder
        relative_parent = file_path.parent.relative_to(DATASET_PATH)

        if str(relative_parent) == ".":
            folder_name = "ROOT"
        else:
            folder_name = str(relative_parent)

        folder_counts[folder_name] += 1

        # Detect possible train/val/test folder
        for part in file_path.parts:
            if part.lower() in SPLIT_NAMES:
                split_counts[part.lower()] += 1
                break

        # Check image integrity and dimensions
        try:

            with Image.open(file_path) as img:
                img.verify()

            # Reopen because verify() invalidates the image object
            with Image.open(file_path) as img:
                width, height = img.size
                mode = img.mode

            image_dimensions[(width, height)] += 1
            image_modes[mode] += 1

        except Exception as e:

            corrupted_images.append({
                "path": str(file_path),
                "error": str(e)
            })

    # --------------------------------------------------------
    # ANNOTATION
    # --------------------------------------------------------

    elif extension in ANNOTATION_EXTENSIONS:

        annotation_count += 1
        annotation_extensions[extension] += 1
        total_annotation_size += file_size

        if len(annotation_samples) < 20:
            annotation_samples.append(str(file_path))

        relative_parent = file_path.parent.relative_to(DATASET_PATH)

        if str(relative_parent) == ".":
            folder_name = "ROOT"
        else:
            folder_name = str(relative_parent)

        folder_counts[folder_name] += 1

    # --------------------------------------------------------
    # OTHER
    # --------------------------------------------------------

    else:

        other_count += 1
        other_extensions[extension if extension else "[NO_EXTENSION]"] += 1
        total_other_size += file_size


# ============================================================
# DATASET SIZE
# ============================================================

total_size = (
    total_image_size
    + total_annotation_size
    + total_other_size
)


def bytes_to_gb(size):
    return round(size / (1024 ** 3), 3)


def bytes_to_mb(size):
    return round(size / (1024 ** 2), 3)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(f"\nTotal files           : {len(all_files)}")
print(f"Images                : {image_count}")
print(f"Annotations           : {annotation_count}")
print(f"Other files           : {other_count}")

print(f"\nTotal dataset size    : {bytes_to_gb(total_size)} GB")
print(f"Image size            : {bytes_to_gb(total_image_size)} GB")
print(f"Annotation size       : {bytes_to_mb(total_annotation_size)} MB")
print(f"Other files size      : {bytes_to_mb(total_other_size)} MB")


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

print("\n" + "-" * 70)
print("IMAGE EXTENSIONS")
print("-" * 70)

for ext, count in image_extensions.most_common():
    print(f"{ext:<10} : {count}")


# ============================================================
# ANNOTATION EXTENSIONS
# ============================================================

print("\n" + "-" * 70)
print("ANNOTATION EXTENSIONS")
print("-" * 70)

if annotation_extensions:

    for ext, count in annotation_extensions.most_common():
        print(f"{ext:<10} : {count}")

else:
    print("No candidate annotation files found.")


# ============================================================
# OTHER FILE TYPES
# ============================================================

print("\n" + "-" * 70)
print("OTHER FILE TYPES")
print("-" * 70)

for ext, count in other_extensions.most_common(20):
    print(f"{ext:<15} : {count}")


# ============================================================
# IMAGE DIMENSIONS
# ============================================================

print("\n" + "-" * 70)
print("IMAGE DIMENSIONS")
print("-" * 70)

for dimensions, count in image_dimensions.most_common(20):
    width, height = dimensions
    print(f"{width} x {height:<8} : {count}")


# ============================================================
# IMAGE MODES
# ============================================================

print("\n" + "-" * 70)
print("IMAGE MODES")
print("-" * 70)

for mode, count in image_modes.most_common():
    print(f"{mode:<10} : {count}")


# ============================================================
# POSSIBLE SPLITS
# ============================================================

print("\n" + "-" * 70)
print("POSSIBLE DATASET SPLITS")
print("-" * 70)

if split_counts:

    for split, count in split_counts.most_common():
        print(f"{split:<15} : {count}")

else:
    print("No train/validation/test folder names detected.")


# ============================================================
# CORRUPTED IMAGES
# ============================================================

print("\n" + "-" * 70)
print("IMAGE INTEGRITY")
print("-" * 70)

print(f"Corrupted/unreadable images : {len(corrupted_images)}")

if corrupted_images:

    print("\nFirst 10 problematic images:")

    for item in corrupted_images[:10]:
        print(f"\n{item['path']}")
        print(f"Error: {item['error']}")


# ============================================================
# SAMPLE IMAGE FILES
# ============================================================

print("\n" + "-" * 70)
print("SAMPLE IMAGE FILES")
print("-" * 70)

for path in image_samples:
    print(path)


# ============================================================
# SAMPLE ANNOTATION FILES
# ============================================================

print("\n" + "-" * 70)
print("SAMPLE ANNOTATION FILES")
print("-" * 70)

for path in annotation_samples:
    print(path)


# ============================================================
# SAVE FILE INVENTORY CSV
# ============================================================

inventory_csv = REPORTS_DIR / "stage1_file_inventory.csv"

with open(
    inventory_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "relative_path",
        "file_name",
        "extension",
        "file_type",
        "size_bytes",
        "size_mb"
    ])

    for file_path in all_files:

        extension = file_path.suffix.lower()

        try:
            size = file_path.stat().st_size
        except OSError:
            size = 0

        if extension in IMAGE_EXTENSIONS:
            file_type = "image"

        elif extension in ANNOTATION_EXTENSIONS:
            file_type = "annotation"

        else:
            file_type = "other"

        writer.writerow([
            str(file_path.relative_to(DATASET_PATH)),
            file_path.name,
            extension,
            file_type,
            size,
            round(size / (1024 ** 2), 3)
        ])


# ============================================================
# SAVE FOLDER SUMMARY CSV
# ============================================================

folder_csv = REPORTS_DIR / "stage1_folder_summary.csv"

with open(
    folder_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "folder",
        "file_count"
    ])

    for folder, count in folder_counts.most_common():
        writer.writerow([
            folder,
            count
        ])


# ============================================================
# SAVE CORRUPTED IMAGE REPORT
# ============================================================

corrupt_csv = REPORTS_DIR / "stage1_corrupted_images.csv"

with open(
    corrupt_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "image_path",
        "error"
    ])

    for item in corrupted_images:
        writer.writerow([
            item["path"],
            item["error"]
        ])


# ============================================================
# SAVE SUMMARY JSON
# ============================================================

summary = {
    "dataset_path": str(DATASET_PATH),

    "total_files": len(all_files),

    "images": {
        "count": image_count,
        "total_size_gb": bytes_to_gb(total_image_size),
        "extensions": dict(image_extensions),
        "dimensions": {
            f"{width}x{height}": count
            for (width, height), count
            in image_dimensions.items()
        },
        "modes": dict(image_modes)
    },

    "annotations": {
        "count": annotation_count,
        "total_size_mb": bytes_to_mb(total_annotation_size),
        "extensions": dict(annotation_extensions)
    },

    "other_files": {
        "count": other_count,
        "total_size_mb": bytes_to_mb(total_other_size),
        "extensions": dict(other_extensions)
    },

    "possible_splits": dict(split_counts),

    "corrupted_images": len(corrupted_images),

    "total_dataset_size_gb": bytes_to_gb(total_size)
}


summary_json = REPORTS_DIR / "stage1_dataset_summary.json"

with open(
    summary_json,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("STAGE 1 COMPLETED")
print("=" * 70)

print("\nGenerated reports:")

print(f"1. {inventory_csv}")
print(f"2. {folder_csv}")
print(f"3. {corrupt_csv}")
print(f"4. {summary_json}")

print("\nIMPORTANT:")
print("No files inside the original CSDD dataset were modified.")
print("No files were deleted.")
print("No preprocessing has been performed yet.")

print("\nNext step will be decided from these actual results.")
print("=" * 70)