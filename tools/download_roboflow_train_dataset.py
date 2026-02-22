# -*- coding: utf-8 -*-
"""
roadscan_dataset_prep.py  —  RoadScan AI Dataset Builder
=========================================================
Pipeline:
  1. Download from Roboflow (verified working slugs)
  2. Clean  — reject corrupt headers + undersized images
  3. Balance — undersample or oversample to TARGET_PER_CLASS
  4. Split  — strict 80 / 10 / 10
  5. Write  — dataset/{train,val,test}/{class}/{image}.jpg
  6. Output — metadata.csv + dataset.yaml

Install:
    pip install roboflow pillow scikit-learn tqdm pyyaml
"""

import csv, random, shutil, struct, zlib
from pathlib import Path
from collections import defaultdict

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ══════════════════════════════════════════════════════════════════
# CONFIG — edit here only
# ══════════════════════════════════════════════════════════════════

ROBOFLOW_API_KEY = "9qzVKhH6gRnn2YH0LWpi"   # roboflow.com → Settings → API

# Verified working slugs (confirmed from run logs).
# Tuple: (workspace_slug, project_slug, version_int, export_format)
#   "yolov8"  = object-detection project (most on Universe)
#   "folder"  = classification project only
ROBOFLOW_SOURCES = {

    "potholes": [
        ("smartathon",
         "new-pothole-detection",                              1, "yolov8"),
        ("indian-institute-of-technology-madras-xamot",
         "pothole-detection-huf2x",                           1, "yolov8"),
    ],

    "cracked_pavement": [
        ("new-workspace-kj87b",
         "road-damage-detection-iicdh",                       1, "yolov8"),
        ("road-crack-project",
         "road-crack-detection-combined-dataset",             1, "yolov8"),
        ("crack-detection-aj3ge",
         "road-damage-detection-apxtk",                       1, "yolov8"),
    ],

    "road_debris_obstruction": [
        ("magic",
         "road-debris",                                       1, "yolov8"),
        # Only 999 images confirmed — balancer will oversample to TARGET
    ],

    "broken_road_signs": [
        ("matyworkspace",
         "damaged-traffic-signs",                             1, "yolov8"),
        ("roboflow-100",
         "road-signs-6ih4y",                                  1, "yolov8"),
        # dropped: usmanchaudhry622-gmail-com/traffic-and-road-signs
        # reason: 15k general sign images, not specifically damaged — too noisy
    ],

    "faded_lane_markings": [
        ("lane-keeping",
         "lane_keeping_1",                                    1, "yolov8"),
        ("road-marking-detection",
         "road-marking-iflo2",                                1, "yolov8"),
        ("wzhen-vt-edu",
         "pavement-cracks-2wi3m",                            1, "yolov8"),
    ],

    "normal_road": [
        ("tobias-price-lane-detection-solid-and-dashed",
         "lane-detection-segmentation-edyqp",                 1, "yolov8"),
        # Only 130 images confirmed — balancer will oversample to TARGET
        # Add more slugs here if you find suitable normal-road datasets
    ],
}

TARGET_PER_CLASS = 1200   # after balancing: ~960 train / 120 val / 120 test
MIN_WIDTH        = 64
MIN_HEIGHT       = 64
SEED             = 42
OUTPUT_DIR       = Path("dataset")
RAW_DIR          = Path("_raw")

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(SEED)


# ══════════════════════════════════════════════════════════════════
# STEP 1 — DOWNLOAD
# ══════════════════════════════════════════════════════════════════

def download_all(raw_dir):
    """
    Downloads every Roboflow source and pools images into raw_dir/<class>/.
    Uses yolov8 format for OD projects — label .txt files are skipped,
    only images are copied. Roboflow's split structure is ignored because
    we create our own 80/10/10 split in step 4.
    """
    from roboflow import Roboflow
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    collected = defaultdict(list)

    for cls_name, sources in ROBOFLOW_SOURCES.items():
        cls_dir = raw_dir / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)

        for workspace, project_slug, version, fmt in sources:
            tmp = raw_dir / f"_tmp_{project_slug}_v{version}"
            try:
                print(f"  down  {workspace}/{project_slug} v{version}  ->  [{cls_name}]")
                proj = rf.workspace(workspace).project(project_slug)
                proj.version(version).download(fmt, location=str(tmp), overwrite=True)

                for img_path in tmp.rglob("*"):
                    if img_path.suffix.lower() not in VALID_EXTS:
                        continue
                    if "labels" in img_path.parts:
                        continue
                    dest = cls_dir / f"{project_slug}_{img_path.name}"
                    shutil.copy2(img_path, dest)
                    collected[cls_name].append(dest)

                shutil.rmtree(tmp, ignore_errors=True)
                print(f"       {len(collected[cls_name])} images pooled for [{cls_name}]")

            except Exception as e:
                print(f"  skip  {workspace}/{project_slug}: {e}")
                shutil.rmtree(tmp, ignore_errors=True)

    return dict(collected)


# ══════════════════════════════════════════════════════════════════
# STEP 2 — CLEAN
# ══════════════════════════════════════════════════════════════════

def is_valid(path):
    """
    Two-stage check:
      1. PIL verify()  — catches truncated files and corrupt headers
      2. Dimension gate — drops thumbnails / accidentally tiny images
    """
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:   # re-open; verify() closes the file handle
            w, h = img.size
            return w >= MIN_WIDTH and h >= MIN_HEIGHT
    except (UnidentifiedImageError, OSError, struct.error, zlib.error, SyntaxError):
        return False


def clean(collected):
    cleaned = {}
    for cls, paths in collected.items():
        valid = [p for p in tqdm(paths, desc=f"  clean [{cls}]") if is_valid(p)]
        print(f"  [{cls}]  {len(paths)} raw  ->  {len(valid)} valid")
        cleaned[cls] = valid
    return cleaned


# ══════════════════════════════════════════════════════════════════
# STEP 3 — BALANCE
# ══════════════════════════════════════════════════════════════════

def balance(cleaned):
    """
    Equalise every class to exactly TARGET_PER_CLASS images.

      n > TARGET  ->  random.sample()   undersample, no duplicates
      n < TARGET  ->  random.choices()  oversample with replacement
      n < 100     ->  skip (too thin; add more Roboflow sources)

    Oversampled images get unique output filenames via index-based
    naming in split_and_write(), so duplicate source paths are fine.
    """
    balanced = {}
    for cls, paths in cleaned.items():
        n = len(paths)
        if n < 100:
            print(f"  WARN  [{cls}] only {n} images after cleaning — skipping.")
            continue
        if n >= TARGET_PER_CLASS:
            sampled = random.sample(paths, TARGET_PER_CLASS)
            print(f"  down  [{cls}]  {n} -> {TARGET_PER_CLASS}  (undersample)")
        else:
            extra   = random.choices(paths, k=TARGET_PER_CLASS - n)
            sampled = paths + extra
            random.shuffle(sampled)
            print(f"  up    [{cls}]  {n} -> {TARGET_PER_CLASS}  (oversample)")
        balanced[cls] = sampled
    return balanced


# ══════════════════════════════════════════════════════════════════
# STEP 4 + 5 — SPLIT & WRITE
# ══════════════════════════════════════════════════════════════════

def split_and_write(balanced, out):
    """
    Stratified 80/10/10 per class, then convert every image to
    RGB JPEG and write to dataset/{train,val,test}/{class}/{name}.jpg
    """
    rows = []

    for cls, paths in balanced.items():
        train, temp = train_test_split(paths, test_size=0.20, random_state=SEED)
        val,  test  = train_test_split(temp,  test_size=0.50, random_state=SEED)

        for split, split_paths in [("train", train), ("val", val), ("test", test)]:
            dst = out / split / cls
            dst.mkdir(parents=True, exist_ok=True)

            for idx, src in enumerate(
                tqdm(split_paths, desc=f"  write [{split}/{cls}]", leave=False)
            ):
                out_name = f"{cls}_{idx:05d}.jpg"
                out_path = dst / out_name
                try:
                    with Image.open(src) as img:
                        img.convert("RGB").save(out_path, "JPEG", quality=92)
                except Exception:
                    continue

                # Dummy GPS — continental US bounding box for Maps integration testing
                lat = round(random.uniform(25.0, 49.0), 6)
                lon = round(random.uniform(-125.0, -66.0), 6)
                rows.append({
                    "filename":  str(out_path.relative_to(out)),
                    "label":     cls,
                    "split":     split,
                    "latitude":  lat,
                    "longitude": lon,
                })

        print(f"  ok    [{cls}]  train={len(train)}  val={len(val)}  test={len(test)}")

    return rows


# ══════════════════════════════════════════════════════════════════
# STEP 6 — METADATA CSV + DATASET YAML
# ══════════════════════════════════════════════════════════════════

def write_metadata(rows, out):
    path = out / "metadata.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "label", "split", "latitude", "longitude"])
        w.writeheader()
        w.writerows(rows)
    print(f"  metadata.csv  ->  {path}  ({len(rows)} rows)")


def write_yaml(out, classes):
    import yaml
    cfg = {
        "path":  str(out.resolve()),
        "train": "train",
        "val":   "val",
        "test":  "test",
        "nc":    len(classes),
        "names": classes,
    }
    with open(out / "dataset.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  dataset.yaml  ->  {out / 'dataset.yaml'}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  RoadScan AI — Dataset Preparation")
    print("=" * 55)

    RAW_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\n[1/5]  Downloading from Roboflow...")
    collected = download_all(RAW_DIR)

    print("\n[2/5]  Cleaning (corrupt + undersized)...")
    cleaned = clean(collected)

    print("\n[3/5]  Balancing classes...")
    balanced = balance(cleaned)

    if not balanced:
        print("ERROR  No classes survived. Check API key and Roboflow slugs.")
        return

    print("\n[4/5]  Splitting 80/10/10 and writing dataset...")
    rows = split_and_write(balanced, OUTPUT_DIR)

    print("\n[5/5]  Writing metadata...")
    write_metadata(rows, OUTPUT_DIR)
    write_yaml(OUTPUT_DIR, list(balanced.keys()))

    shutil.rmtree(RAW_DIR, ignore_errors=True)

    print("\n" + "=" * 55)
    print("  DONE")
    print(f"\n  {OUTPUT_DIR.resolve()}/")
    for split in ("train", "val", "test"):
        d = OUTPUT_DIR / split
        if d.exists():
            n = sum(len(list(c.iterdir())) for c in d.iterdir() if c.is_dir())
            print(f"    {split}/   {n} images")
    print("    metadata.csv")
    print("    dataset.yaml")
    print("=" * 55)
    print(f"\nIn roadscan_colab_2.py set:  DATA_DIR = '{OUTPUT_DIR.resolve()}'\n")


if __name__ == "__main__":
    main()

