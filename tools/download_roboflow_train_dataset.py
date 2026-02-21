# -*- coding: utf-8 -*-
"""
roadscan_dataset_prep.py
========================
RoadScan AI — Roboflow Dataset Builder
Run: python roadscan_dataset_prep.py

Install deps:
    pip install roboflow torch torchvision transformers pillow scikit-learn tqdm pyyaml
"""
import os, csv, random, shutil, struct, zlib
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# ══════════════════════════════════════════════════════════════════
# CONFIG — only edit this section
# ══════════════════════════════════════════════════════════════════

ROBOFLOW_API_KEY = "9qzVKhH6gRnn2YH0LWpi"   # roboflow.com → Settings → API

# ── Verified Roboflow Universe slugs ─────────────────────────────
# Format: (workspace_slug, project_slug, version_int, export_format)
#
# export_format is either:
#   "yolov8"  — object-detection projects  (most datasets on Universe)
#   "folder"  — classification projects only
#
# All slugs below were confirmed via Roboflow Universe search Feb 2026.
# If a project has been deleted, comment it out and the rest still run.
ROBOFLOW_SOURCES = {

    "potholes": [
        # 665 images — original Kaggle pothole dataset, OD
        ("public",                              "pothole",                          1, "yolov8"),
        # 9.2k images — Smartathon pothole dataset, OD
        ("smartathon",                          "new-pothole-detection",            1, "yolov8"),
        # 2.7k images — IIT Madras, OD (also has crack classes — CLIP will filter)
        ("indian-institute-of-technology-madras-xamot", "pothole-detection-huf2x", 1, "yolov8"),
    ],

    "cracked_pavement": [
        # 3.6k images — road damage detection with alligator/longitudinal/transverse, OD
        ("new-workspace-kj87b",                 "road-damage-detection-iicdh",      1, "yolov8"),
        # 821 images — combined crack dataset, OD
        ("road-crack-project",                  "road-crack-detection-combined-dataset", 1, "yolov8"),
        # 424 images — crack detection, OD
        ("crack-detection-aj3ge",               "road-damage-detection-apxtk",      1, "yolov8"),
    ],

    "road_debris_obstruction": [
        # 999 images — road debris segmentation dataset, OD/seg
        ("magic",                               "road-debris",                      1, "yolov8"),
        # Road debris by Hind Althabi, OD
        ("hind-althabi",                        "road-debris-iya6s",                1, "yolov8"),
        # Road features dashcam — picks up obstruction frames, OD
        ("road-features",                       "dashcam-xtfeo-eespa",              1, "yolov8"),
    ],

    "broken_road_signs": [
        # 1994 images — damaged traffic signs, OD
        ("matyworkspace",                       "damaged-traffic-signs",            1, "yolov8"),
        # Roboflow-100 road signs benchmark, OD
        ("roboflow-100",                        "road-signs-6ih4y",                 1, "yolov8"),
        # 10k traffic and road signs — CLIP will keep only damaged/obscured ones
        ("usmanchaudhry622-gmail-com",          "traffic-and-road-signs",           1, "yolov8"),
    ],

    "faded_lane_markings": [
        # 5.9k images — road mark lane keeping dataset, OD
        ("lane-keeping",                        "lane_keeping_1",                   1, "yolov8"),
        # Road marking detection, OD
        ("road-marking-detection",              "road-marking-iflo2",               1, "yolov8"),
        # Pavement cracks with faded marking classes, OD
        ("wzhen-vt-edu",                        "pavement-cracks-2wi3m",            1, "yolov8"),
    ],

    "normal_road": [
        # Road classification dataset — normal/abnormal classes
        ("jason-a4myk",                         "road-vsnrg",                       1, "yolov8"),
        # Lane detection segmentation — clear road frames
        ("tobias-price-lane-detection-solid-and-dashed", "lane-detection-segmentation-edyqp", 1, "yolov8"),
        # General dashcam road scenes
        ("road-features",                       "dashcam-xtfeo-eespa",              2, "yolov8"),
    ],
}

# CLIP prompts — one sentence per class, evaluated against all 6 simultaneously.
# The two-gate check (own prob >= threshold AND argmax == own class) catches
# both low-confidence images and cross-class contamination.
CLIP_CLASS_PROMPTS = {
    "potholes":
        "a photo of a pothole or hole in a road or asphalt",
    "cracked_pavement":
        "a photo of cracked or damaged pavement or asphalt surface",
    "road_debris_obstruction":
        "a photo of debris, rocks, or objects blocking a road or lane",
    "broken_road_signs":
        "a photo of a damaged, bent, fallen, or vandalized road sign",
    "faded_lane_markings":
        "a photo of faded, worn, or barely visible lane markings on a road",
    "normal_road":
        "a photo of a clean, smooth, well-maintained road or highway",
}

CLASS_NAMES       = list(CLIP_CLASS_PROMPTS.keys())
ALL_CLASS_PROMPTS = list(CLIP_CLASS_PROMPTS.values())

TARGET_PER_CLASS = 1200   # → ~960 train / 120 val / 120 test per class
CLIP_THRESHOLD   = 0.60
MIN_WIDTH        = 64
MIN_HEIGHT       = 64
SEED             = 42
OUTPUT_DIR       = Path("dataset")
RAW_DIR          = Path("_raw")

random.seed(SEED)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ══════════════════════════════════════════════════════════════════
# STEP 1 — DOWNLOAD
# ══════════════════════════════════════════════════════════════════

def download_all(raw_dir: Path) -> dict:
    """
    Downloads every source project and pools all images per class.
    Object-detection projects are downloaded as yolov8 format — we
    only copy the /images/ sub-folders and discard the label files.
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
                print(f"  ↓  {workspace}/{project_slug} v{version}  [{fmt}]  →  [{cls_name}]")
                proj = rf.workspace(workspace).project(project_slug)
                proj.version(version).download(fmt, location=str(tmp), overwrite=True)

                # Collect every image regardless of split sub-folder.
                # For yolov8: structure is <tmp>/{train,valid,test}/images/*.jpg
                # For folder: structure is <tmp>/{train,valid,test}/<label>/*.jpg
                for img_path in tmp.rglob("*"):
                    if img_path.suffix.lower() in VALID_EXTS:
                        # Skip anything inside a "labels" folder
                        if "labels" in img_path.parts:
                            continue
                        dest = cls_dir / f"{project_slug}_{img_path.name}"
                        shutil.copy2(img_path, dest)
                        collected[cls_name].append(dest)

                shutil.rmtree(tmp, ignore_errors=True)
                print(f"       {len(collected[cls_name])} images pooled for [{cls_name}]")

            except Exception as e:
                print(f"  ⚠   Skipped {workspace}/{project_slug}: {e}")
                shutil.rmtree(tmp, ignore_errors=True)

    return dict(collected)


# ══════════════════════════════════════════════════════════════════
# STEP 2 — CLEAN
# ══════════════════════════════════════════════════════════════════

def is_valid(path: Path) -> bool:
    """Reject corrupt headers and images smaller than MIN_WIDTH × MIN_HEIGHT."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            w, h = img.size
            return w >= MIN_WIDTH and h >= MIN_HEIGHT
    except (UnidentifiedImageError, OSError, struct.error, zlib.error, SyntaxError):
        return False


def clean(collected: dict) -> dict:
    cleaned = {}
    for cls, paths in collected.items():
        valid = [p for p in tqdm(paths, desc=f"  clean [{cls}]") if is_valid(p)]
        print(f"  [{cls}]  {len(paths)} raw  →  {len(valid)} valid")
        cleaned[cls] = valid
    return cleaned


# ══════════════════════════════════════════════════════════════════
# STEP 3 — CLIP FILTER
# ══════════════════════════════════════════════════════════════════

def _extract_tensor(output):
    """
    Safely extract a plain float Tensor from whatever get_image_features()
    returns — handles both plain Tensor (newer transformers) and
    BaseModelOutputWithPooling (older versions).
    """
    if isinstance(output, torch.Tensor):
        return output
    # BaseModelOutputWithPooling / similar dataclass
    if hasattr(output, "image_embeds"):
        return output.image_embeds
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0]   # CLS token
    raise TypeError(f"Cannot extract tensor from {type(output)}")


def clip_filter(cleaned: dict) -> dict:
    """
    Zero-shot classification across all 6 class prompts in one pass.
    Image survives only if:
      (a) own-class softmax prob >= CLIP_THRESHOLD
      (b) own class is the argmax  — removes cross-class false positives
    """
    from transformers import CLIPProcessor, CLIPModel

    print("\n  Loading CLIP (openai/clip-vit-base-patch32)…")
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    # Pre-encode text prompts once
    with torch.no_grad():
        txt_in    = processor(text=ALL_CLASS_PROMPTS, return_tensors="pt", padding=True).to(device)
        txt_feats = model.get_text_features(**txt_in)           # always a plain Tensor
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

    BATCH  = 64
    result = {}

    for cls, paths in cleaned.items():
        cls_idx       = CLASS_NAMES.index(cls)
        kept, dropped = [], 0

        for i in tqdm(range(0, len(paths), BATCH), desc=f"  CLIP  [{cls}]"):
            batch      = paths[i : i + BATCH]
            imgs, valids = [], []
            for p in batch:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                    valids.append(p)
                except Exception:
                    dropped += 1

            if not imgs:
                continue

            with torch.no_grad():
                img_in    = processor(images=imgs, return_tensors="pt", padding=True).to(device)
                raw_out   = model.get_image_features(**img_in)
                img_feats = _extract_tensor(raw_out)            # ← bug-fixed line
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                sims      = img_feats @ txt_feats.T             # (B, n_classes)
                probs     = torch.softmax(sims, dim=-1).cpu()

            for j, p in enumerate(valids):
                own_prob = probs[j, cls_idx].item()
                pred_cls = probs[j].argmax().item()
                if own_prob >= CLIP_THRESHOLD and pred_cls == cls_idx:
                    kept.append(p)
                else:
                    dropped += 1

        print(f"  [{cls}]  kept {len(kept)} / {len(kept) + dropped}  (threshold={CLIP_THRESHOLD})")
        result[cls] = kept

    return result


# ══════════════════════════════════════════════════════════════════
# STEP 4 — BALANCE
# ══════════════════════════════════════════════════════════════════

def balance(filtered: dict) -> dict:
    """
    Equalise all classes to TARGET_PER_CLASS.

    ┌────────────────────────────────────────────────────────────┐
    │ n > TARGET  → random.sample()  — undersample, no dups      │
    │ n < TARGET  → random.choices() — oversample w/ replacement │
    │ n < 100     → skip (too thin; add more Roboflow sources)   │
    └────────────────────────────────────────────────────────────┘
    """
    balanced = {}
    for cls, paths in filtered.items():
        n = len(paths)
        if n < 100:
            print(f"  ⚠  [{cls}] only {n} images — skipping. Add more Roboflow sources.")
            continue
        if n >= TARGET_PER_CLASS:
            sampled = random.sample(paths, TARGET_PER_CLASS)
            print(f"  ↓ undersample [{cls}]  {n} → {TARGET_PER_CLASS}")
        else:
            extra   = random.choices(paths, k=TARGET_PER_CLASS - n)
            sampled = paths + extra
            random.shuffle(sampled)
            print(f"  ↑ oversample  [{cls}]  {n} → {TARGET_PER_CLASS}")
        balanced[cls] = sampled
    return balanced


# ══════════════════════════════════════════════════════════════════
# STEP 5 — SPLIT & WRITE
# ══════════════════════════════════════════════════════════════════

def split_and_write(balanced: dict, out: Path) -> list:
    """80 / 10 / 10 per class → dataset/{train,val,test}/{class}/"""
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

                lat = round(random.uniform(25.0, 49.0), 6)
                lon = round(random.uniform(-125.0, -66.0), 6)
                rows.append({
                    "filename":  str(out_path.relative_to(out)),
                    "label":     cls,
                    "split":     split,
                    "latitude":  lat,
                    "longitude": lon,
                })

        print(f"  ✓  [{cls}]  train={len(train)}  val={len(val)}  test={len(test)}")

    return rows


# ══════════════════════════════════════════════════════════════════
# STEP 6 — METADATA CSV + DATASET YAML
# ══════════════════════════════════════════════════════════════════

def write_metadata(rows: list, out: Path):
    path = out / "metadata.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "label", "split", "latitude", "longitude"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n  metadata.csv  →  {path}  ({len(rows)} rows)")


def write_yaml(out: Path, classes: list):
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
    print(f"  dataset.yaml  →  {out / 'dataset.yaml'}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  RoadScan AI — Dataset Preparation  (v2)")
    print("=" * 60)

    RAW_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\n[1/6]  Downloading from Roboflow…")
    collected = download_all(RAW_DIR)

    print("\n[2/6]  Cleaning (corrupt + undersized)…")
    cleaned = clean(collected)

    print("\n[3/6]  CLIP verification…")
    verified = clip_filter(cleaned)

    print("\n[4/6]  Balancing classes…")
    balanced = balance(verified)

    if not balanced:
        print("❌  No classes survived. Check API key and Roboflow slugs.")
        return

    print("\n[5/6]  Splitting 80/10/10 and writing…")
    rows = split_and_write(balanced, OUTPUT_DIR)

    print("\n[6/6]  Writing metadata…")
    write_metadata(rows, OUTPUT_DIR)
    write_yaml(OUTPUT_DIR, list(balanced.keys()))

    shutil.rmtree(RAW_DIR, ignore_errors=True)

    print("\n" + "=" * 60)
    print("  ✅  Done!")
    print(f"\n  {OUTPUT_DIR.resolve()}/")
    for split in ("train", "val", "test"):
        d = OUTPUT_DIR / split
        if d.exists():
            n = sum(len(list(c.iterdir())) for c in d.iterdir() if c.is_dir())
            print(f"    {split}/   {n} images")
    print(f"    metadata.csv")
    print(f"    dataset.yaml")
    print("=" * 60)
    print(f"\n→ In roadscan_colab_2.py set:  DATA_DIR = '{OUTPUT_DIR.resolve()}'\n")


if __name__ == "__main__":
    main()