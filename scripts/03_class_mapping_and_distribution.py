from pathlib import Path
from collections import Counter
from PIL import Image
import csv, json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "CSDD_seg"
IMAGE_ROOT = DATASET_PATH / "img"
MASK_ROOT = DATASET_PATH / "ground_truth"
REPORTS_DIR = PROJECT_ROOT / "reports"
SPLITS = ["train", "val", "test"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
REPORTS_DIR.mkdir(exist_ok=True)


def find_docs():
    docs = []
    for p in DATASET_PATH.rglob("*"):
        if p.is_file() and ("readme" in p.name.lower() or p.suffix.lower() in {".txt", ".md"}):
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            low = text.lower()
            if all(x in low for x in ["scratch", "spot", "rust"]):
                docs.append({"file": str(p), "snippet": text[:5000]})
    return docs


def analyze_mask(mask):
    # CSDD masks are P-mode. getcolors is much faster than Counter(mask.getdata()).
    colors = mask.getcolors(maxcolors=256)
    if colors is None:
        return None
    return Counter(int(value) for count, value in colors for _ in range(0)) if False else Counter({int(v): int(c) for c, v in colors})


def palette(mask):
    if mask.mode != "P" or not mask.getpalette():
        return {}
    pal = mask.getpalette()
    return {str(i): [pal[i*3], pal[i*3+1], pal[i*3+2]] for i in range(4) if i*3+2 < len(pal)}


def main():
    print("=" * 70)
    print("STAGE 3 - CLASS MAPPING AND CLASS DISTRIBUTION")
    print("=" * 70)
    print(f"Dataset: {DATASET_PATH}\n")

    docs = find_docs()
    if docs:
        print("Documentation containing Scratch / Spot / Rust found:")
        for d in docs:
            print(" ", d["file"])
    else:
        print("No local documentation explicitly confirms the 1/2/3 class mapping.")
        print("We will NOT guess the mapping.")
    print()

    global_counts = Counter()
    global_images = Counter()
    rows = []
    palettes = {}

    for split in SPLITS:
        print("=" * 70)
        print(f"PROCESSING: {split.upper()}")
        print("=" * 70)
        idir, mdir = IMAGE_ROOT / split, MASK_ROOT / split
        images = sorted(p for p in idir.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
        masks = {}
        for p in mdir.iterdir():
            if p.is_file() and p.suffix.lower() in MASK_EXTS:
                s = p.stem.lower()
                if s.endswith("_mask"): s = s[:-5]
                masks[s] = p

        split_counts = Counter()
        split_images = Counter()
        total = 0

        for n, img in enumerate(images, 1):
            mp = masks.get(img.stem.lower())
            if not mp: continue
            try:
                with Image.open(mp) as m:
                    counts = analyze_mask(m)
                    if counts is None:
                        raise ValueError("more than 256 unique mask values")
                    pal = palette(m)
                    palettes.update(pal)
                    total += sum(counts.values())
                    for v in [0,1,2,3]:
                        c = counts.get(v, 0)
                        split_counts[v] += c
                        global_counts[v] += c
                        if v and c > 0:
                            split_images[v] += 1
                            global_images[v] += 1
                    rows.append({"split":split,"image":img.name,"mask":mp.name,**{f"value_{v}_pixels":counts.get(v,0) for v in [0,1,2,3]}})
            except Exception as e:
                rows.append({"split":split,"image":img.name,"mask":mp.name,"error":str(e)})
            if n % 250 == 0 or n == len(images): print(f"Processed {n}/{len(images)} masks...")

        print("\n--- Split Result ---")
        print(f"Total pixels      : {total:,}")
        for v in [0,1,2,3]: print(f"Pixel value {v}     : {split_counts[v]:,}")
        print(f"Images containing 1/2/3: {split_images[1]} / {split_images[2]} / {split_images[3]}\n")

    total_pixels = sum(global_counts.values())
    defect_pixels = sum(global_counts[v] for v in [1,2,3])
    summary = {
        "mapping_status": "NOT_CONFIRMED" if not docs else "DOCUMENTATION_FOUND_REQUIRES_REVIEW",
        "documentation_evidence": docs,
        "global_pixel_counts": {str(v): global_counts[v] for v in [0,1,2,3]},
        "global_pixel_percentages": {str(v): (global_counts[v]/total_pixels*100 if total_pixels else 0) for v in [0,1,2,3]},
        "images_containing_each_defect_value": {str(v): global_images[v] for v in [1,2,3]},
        "defect_pixel_percentage": defect_pixels/total_pixels*100 if total_pixels else 0,
        "palette_rgb": palettes,
    }
    jp = REPORTS_DIR / "stage3_class_distribution_summary.json"
    cp = REPORTS_DIR / "stage3_per_image_class_pixels.csv"
    jp.write_text(json.dumps(summary, indent=4), encoding="utf-8")
    with cp.open("w", newline="", encoding="utf-8") as f:
        fields = ["split","image","mask","value_0_pixels","value_1_pixels","value_2_pixels","value_3_pixels","error"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,"") for k in fields})

    print("=" * 70)
    print("GLOBAL RESULT")
    print("=" * 70)
    for v in [0,1,2,3]: print(f"Pixel value {v}: {global_counts[v]:,} pixels ({global_counts[v]/total_pixels*100:.6f}%)")
    print(f"Defect pixels (1+2+3): {defect_pixels:,}")
    print(f"Defect pixel percentage: {defect_pixels/total_pixels*100:.6f}%")
    print("\nImages containing each value:")
    for v in [1,2,3]: print(f"  {v}: {global_images[v]}")
    print("\nPalette RGB:")
    for v, rgb in sorted(palettes.items()): print(f"  {v}: {rgb}")
    print("\nIMPORTANT: Do not assume 1=Scratch, 2=Spot, 3=Rust until mapping is verified.")
    print(f"\nGenerated:\n1. {jp}\n2. {cp}")
    print("Original dataset files were not modified.")
    print("=" * 70)

if __name__ == "__main__": main()
