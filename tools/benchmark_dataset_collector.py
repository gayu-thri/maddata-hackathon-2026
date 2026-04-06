"""
RoadScan AI — Dataset Collection Pipeline
==========================================
Step 0: Class search queries (defined below)
Step 1: Bing image scraping via icrawler (free, no API key)
Step 2: AI verification via transformers zero-shot CLIP (free, local, no API key)
Step 3: De-duplication via perceptual hashing (imagehash)

Install deps:
    pip install icrawler transformers torch torchvision pillow imagehash tqdm

Run:
    python roadscan_collect_dataset.py
"""

import shutil
import warnings
from pathlib import Path

import imagehash
import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from icrawler.builtin import BingImageCrawler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STEP 0 — CONFIG & CLASS SEARCH QUERIES
# ─────────────────────────────────────────────
IMAGES_PER_CLASS    = 10000
SAVE_DIR            = Path("road_dataset_raw")       # raw scraped images
VERIFIED_DIR        = Path("road_dataset_verified")  # after CLIP verification
CLIP_CONFIDENCE     = 0.20   # min probability that image belongs to its class
PHASH_THRESHOLD     = 8      # hamming distance — lower = stricter dedup

CLASS_SEARCH_QUERIES = {
    "potholes": [
        "pothole in asphalt road close up",
        "large pothole on city street",
        "deep pothole highway damage",
        "car hitting pothole road",
        "pothole from dashcam perspective",
        "water filled pothole after rain",
        "urban pothole road surface",
        "rural road pothole damage",
    ],
    "cracked_pavement": [
        "alligator cracking asphalt road",
        "longitudinal crack pavement",
        "transverse crack asphalt",
        "road surface crack close up",
        "cracked road from top view",
        "cracked pavement drone view",
        "severe pavement cracking highway",
    ],
    "road_debris_obstruction": [
        "fallen tree blocking road",
        "rock on highway road hazard",
        "debris on roadway dashcam",
        "construction debris road",
        "object blocking lane road",
        "road obstruction accident scene",
        "storm debris on road",
    ],
    "broken_road_signs": [
        "damaged traffic sign pole",
        "bent stop sign roadside",
        "fallen road sign on street",
        "vandalized traffic sign",
        "broken street sign urban",
        "rusted damaged road sign",
    ],
    "faded_lane_markings": [
        "faded lane lines highway",
        "worn road markings asphalt",
        "barely visible lane markings",
        "old pavement paint faded",
        "night road faded lane markings",
        "rainy road faded lines",
    ],
    "normal_road": [
        "smooth asphalt road daytime",
        "clean highway with clear lane markings",
        "well maintained city road",
        "newly paved road surface",
        "clear road dashcam view",
        "urban road no damage",
    ],
}

# Human-readable CLIP prompts per class (more descriptive = better CLIP accuracy)
CLIP_CLASS_PROMPTS = {
    "potholes":                "a photo of a pothole or hole in a road or asphalt",
    "cracked_pavement":        "a photo of cracked or damaged pavement or asphalt surface",
    "road_debris_obstruction": "a photo of debris, rocks, or objects blocking a road or lane",
    "broken_road_signs":       "a photo of a damaged, bent, fallen, or vandalized road sign",
    "faded_lane_markings":     "a photo of faded, worn, or barely visible lane markings on a road",
    "normal_road":             "a photo of a clean, smooth, well-maintained road or highway",
}

# Negative prompt — acts as a soft reject bucket in CLIP scoring
NEGATIVE_PROMPT = "a photo of something unrelated to roads or streets"


# ─────────────────────────────────────────────
# STEP 1 — BING IMAGE SCRAPING
# ─────────────────────────────────────────────
def scrape_images():
    print("\n" + "=" * 60)
    print("STEP 1 — Bing Image Scraping")
    print("=" * 60)

    for class_name, queries in CLASS_SEARCH_QUERIES.items():
        class_dir = SAVE_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        images_per_query = max(1, IMAGES_PER_CLASS // len(queries))
        print(f"\n[Class] {class_name} — {images_per_query} images/query x {len(queries)} queries")

        for query in queries:
            print(f"  -> Searching: '{query}'")
            try:
                crawler = BingImageCrawler(
                    downloader_threads=4,
                    storage={"root_dir": str(class_dir)},
                    log_level=50,  # suppress icrawler noise (logging.CRITICAL)
                )
                crawler.crawl(
                    keyword=query,
                    max_num=images_per_query,
                    overwrite=False,
                    filters={"type": "photo"},
                )
            except Exception as e:
                print(f"    WARNING: Crawl error for '{query}': {e}")

    total = sum(len(list((SAVE_DIR / c).glob("*"))) for c in CLASS_SEARCH_QUERIES)
    print(f"\n[Step 1 Done] Total raw images scraped: {total}")


# ─────────────────────────────────────────────
# STEP 2 — AI VERIFICATION WITH CLIP
# ─────────────────────────────────────────────
def load_clip():
    print("\n[CLIP] Loading openai/clip-vit-base-patch32 (downloads once, ~350MB)...")
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    print(f"[CLIP] Loaded on {device}")
    return model, processor, device


def verify_image_clip(image, class_name, model, processor, device):
    """
    Returns True if CLIP thinks this image belongs to class_name
    with probability > CLIP_CONFIDENCE (vs. all other class prompts + negative).
    """
    all_prompts  = list(CLIP_CLASS_PROMPTS.values()) + [NEGATIVE_PROMPT]
    target_idx   = list(CLIP_CLASS_PROMPTS.keys()).index(class_name)

    inputs = processor(
        text=all_prompts,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs   = outputs.logits_per_image.softmax(dim=1)[0].cpu()

    class_prob = probs[target_idx].item()
    return class_prob >= CLIP_CONFIDENCE


def verify_dataset():
    print("\n" + "=" * 60)
    print("STEP 2 — AI Verification (CLIP zero-shot, no API key)")
    print("=" * 60)

    model, processor, device = load_clip()
    stats = {}

    for class_name in CLASS_SEARCH_QUERIES:
        raw_class_dir      = SAVE_DIR / class_name
        verified_class_dir = VERIFIED_DIR / class_name
        verified_class_dir.mkdir(parents=True, exist_ok=True)

        image_files = list(raw_class_dir.glob("*"))
        kept     = 0
        rejected = 0

        print(f"\n[Verify] {class_name} — {len(image_files)} raw images")

        for img_path in tqdm(image_files, desc=f"  {class_name}", unit="img"):
            try:
                img = Image.open(img_path).convert("RGB")
            except (UnidentifiedImageError, Exception):
                rejected += 1
                continue

            if verify_image_clip(img, class_name, model, processor, device):
                shutil.copy2(img_path, verified_class_dir / img_path.name)
                kept += 1
            else:
                rejected += 1

        stats[class_name] = {"kept": kept, "rejected": rejected}
        print(f"  Kept: {kept}  |  Rejected: {rejected}")

    print("\n[Step 2 Done] Verification summary:")
    for cls, s in stats.items():
        total = s["kept"] + s["rejected"]
        pct   = (s["kept"] / total * 100) if total else 0
        print(f"  {cls:30s}  kept {s['kept']:>4}/{total}  ({pct:.0f}%)")


# ─────────────────────────────────────────────
# STEP 3 — DE-DUPLICATION (perceptual hash)
# ─────────────────────────────────────────────
def dedup_class(class_dir):
    """
    Remove near-duplicate images within a class using perceptual hashing.
    Returns number of duplicates removed.
    """
    image_files = list(class_dir.glob("*"))
    seen_hashes = []   # list of (phash, path)
    removed = 0

    for img_path in image_files:
        try:
            img   = Image.open(img_path).convert("RGB")
            phash = imagehash.phash(img)
        except Exception:
            img_path.unlink(missing_ok=True)
            removed += 1
            continue

        # Compare against all previously accepted hashes
        is_dup = any(
            abs(phash - seen) <= PHASH_THRESHOLD
            for seen, _ in seen_hashes
        )

        if is_dup:
            img_path.unlink()
            removed += 1
        else:
            seen_hashes.append((phash, img_path))

    return removed


def dedup_dataset():
    print("\n" + "=" * 60)
    print("STEP 3 — De-duplication (perceptual hashing)")
    print("=" * 60)

    for class_name in CLASS_SEARCH_QUERIES:
        class_dir = VERIFIED_DIR / class_name
        if not class_dir.exists():
            print(f"  WARNING: {class_name}: verified dir not found, skipping")
            continue

        before  = len(list(class_dir.glob("*")))
        removed = dedup_class(class_dir)
        after   = before - removed

        print(f"  {class_name:30s}  {before:>4} -> {after:>4}  (removed {removed} duplicates)")

    print("\n[Step 3 Done] Final dataset in:", VERIFIED_DIR)
    print_final_summary()


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
def print_final_summary():
    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)
    total = 0
    for class_name in CLASS_SEARCH_QUERIES:
        class_dir = VERIFIED_DIR / class_name
        count     = len(list(class_dir.glob("*"))) if class_dir.exists() else 0
        total    += count
        bar       = "|" * (count // 10)
        print(f"  {class_name:30s}  {count:>4} images  {bar}")
    print(f"\n  {'TOTAL':30s}  {total:>4} images")
    print(f"\n  Ready for training: python roadscan_train.py --data_dir {VERIFIED_DIR}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("""
+------------------------------------------------+
|      RoadScan AI -- Dataset Pipeline           |
|  Step 0: Class queries defined in config       |
|  Step 1: Bing scrape (icrawler, no API key)    |
|  Step 2: CLIP verification (local, no API key) |
|  Step 3: Perceptual hash deduplication         |
+------------------------------------------------+
    """)

    scrape_images()   # Step 1
    verify_dataset()  # Step 2
    dedup_dataset()   # Step 3