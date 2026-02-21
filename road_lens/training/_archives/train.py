"""
RoadScan AI - Quick Training Script (Modified for YAML + Prefix-Based Dataset)
Fine-tunes EfficientNetV2-S for road hazard classification.

Supports dataset layout:
  data_dir/
  ├── dataset.yaml          ← class names & paths read from here
  ├── train/
  │   ├── images/           ← flat folder, class encoded as filename prefix
  │   └── labels/           ← ignored during training
  └── val/
      ├── images/
      └── labels/

Class names are loaded from dataset.yaml (names: field).
The longest matching class name is found greedily in the filename prefix,
so both single-word ("pothole") and multi-word ("broken_sidewalk") classes
are handled correctly without any manual --prefix_words tuning.

Expected dataset.yaml format:
  names:
    - pothole
    - garbage_overflow
    - broken_streetlight
    - water_leakage
    - broken_sidewalk
    - sinkhole
  nc: 6
  path: /abs/path/to/dataset
  train: train/images
  val: val/images

Usage:
  python roadscan_train_fast.py --data_dir ./datasets/final --epochs 20 --batch_size 16
"""

import argparse
import os
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import EfficientNet_V2_S_Weights
from PIL import Image
import numpy as np
import yaml

IMAGE_SIZE = 224


# ──────────────────────────────────────────────
# YAML LOADER
# ──────────────────────────────────────────────
def load_dataset_yaml(data_dir: Path) -> dict:
    """
    Reads dataset.yaml from data_dir and returns a config dict with:
      - class_names : list[str]   ordered class names
      - train_images: Path        absolute path to train images folder
      - val_images  : Path        absolute path to val images folder
    """
    yaml_path = data_dir / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"dataset.yaml not found at {yaml_path}")

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    class_names = cfg.get("names", [])
    if not class_names:
        raise ValueError("dataset.yaml has no 'names' field or it is empty")

    nc = cfg.get("nc", len(class_names))
    if nc != len(class_names):
        print(f"  ⚠️  dataset.yaml: nc={nc} but {len(class_names)} names found — using names list")

    # Resolve image dirs: prefer absolute path from yaml, fall back to data_dir-relative
    yaml_root = Path(cfg.get("path", str(data_dir)))
    if not yaml_root.is_absolute():
        yaml_root = data_dir / yaml_root

    train_images = yaml_root / cfg.get("train", "train/images")
    val_images   = yaml_root / cfg.get("val",   "val/images")

    print(f"[YAML] Loaded {yaml_path}")
    print(f"[YAML] Classes ({len(class_names)}): {class_names}")
    print(f"[YAML] Train images : {train_images}")
    print(f"[YAML] Val images   : {val_images}")

    return {
        "class_names":  class_names,
        "train_images": train_images,
        "val_images":   val_images,
    }


# ──────────────────────────────────────────────
# DEVICE SELECTION
# ──────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] CUDA — {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Apple MPS (M-series GPU)")
    else:
        device = torch.device("cpu")
        print("[Device] CPU")
    return device


# ──────────────────────────────────────────────
# CUSTOM DATASET — matches class from filename using known class list
# ──────────────────────────────────────────────
class PrefixLabelDataset(Dataset):
    """
    Loads images from a flat `images/` directory.

    Class label is determined by finding the longest class name (from
    class_to_idx) that the filename stem starts with, followed by "_" or
    end-of-prefix. This correctly handles overlapping prefixes like:
        "broken_streetlight_..." → "broken_streetlight"  (not "broken_sidewalk")
        "broken_sidewalk_..."   → "broken_sidewalk"

    Parameters
    ----------
    images_dir   : Path to the flat images folder
    transform    : torchvision transform
    class_to_idx : dict mapping class_name → int index  (required)
    """

    VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, images_dir: Path, transform=None, class_to_idx: dict = None):
        if class_to_idx is None:
            raise ValueError("class_to_idx must be provided (loaded from dataset.yaml)")

        self.images_dir   = images_dir
        self.transform    = transform
        self.class_to_idx = class_to_idx
        self.classes      = sorted(class_to_idx, key=class_to_idx.get)

        # Sort known classes longest-first so greedy match picks the most specific
        self._sorted_classes = sorted(class_to_idx.keys(), key=len, reverse=True)

        self.samples: list[tuple[Path, int]] = []
        skipped = 0

        for p in sorted(images_dir.iterdir()):
            if p.suffix.lower() not in self.VALID_EXTS:
                continue
            cls = self._match_class(p.stem)
            if cls is not None:
                self.samples.append((p, class_to_idx[cls]))
            else:
                skipped += 1

        if skipped:
            print(f"  ⚠️  {images_dir.parent.name}: skipped {skipped} images "
                  f"(prefix didn't match any known class)")

    def _match_class(self, stem: str) -> str | None:
        """Return the longest class name that is a prefix of stem."""
        for cls in self._sorted_classes:          # longest first
            if stem == cls or stem.startswith(cls + "_"):
                return cls
        return None

    @property
    def targets(self):
        return [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ──────────────────────────────────────────────
# DATA TRANSFORMS
# ──────────────────────────────────────────────
def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 24, IMAGE_SIZE + 24)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, val_tf


# ──────────────────────────────────────────────
# DATASET + WEIGHTED SAMPLER
# ──────────────────────────────────────────────
def build_dataloaders(data_dir, batch_size, num_workers):
    train_tf, val_tf = get_transforms()
    data_dir = Path(data_dir)

    # ── Load class config from dataset.yaml ──
    yaml_cfg     = load_dataset_yaml(data_dir)
    class_names  = yaml_cfg["class_names"]
    train_images = yaml_cfg["train_images"]
    val_images   = yaml_cfg["val_images"]

    if not train_images.exists():
        raise FileNotFoundError(f"Train images dir not found: {train_images}")
    if not val_images.exists():
        raise FileNotFoundError(f"Val images dir not found: {val_images}")

    class_to_idx = {name: i for i, name in enumerate(class_names)}

    train_ds = PrefixLabelDataset(train_images, transform=train_tf, class_to_idx=class_to_idx)
    val_ds   = PrefixLabelDataset(val_images,   transform=val_tf,   class_to_idx=class_to_idx)

    print(f"\n[Data] Train: {len(train_ds)} images | Val: {len(val_ds)} images")
    print(f"[Data] Class distribution (train):")

    targets     = torch.tensor(train_ds.targets)
    class_count = torch.bincount(targets, minlength=len(class_names)).float()
    class_count[class_count == 0] = 1

    for i, cls in enumerate(class_names):
        print(f"       [{i}] {cls:25s}: {int(class_count[i]):4d} images")

    # Weighted sampler to handle class imbalance
    weights = 1.0 / class_count[targets]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, class_names


# ──────────────────────────────────────────────
# MODEL — EfficientNetV2-S
# ──────────────────────────────────────────────
def build_model(num_classes, freeze_backbone=True):
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model   = models.efficientnet_v2_s(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(256, num_classes),
    )
    return model


def unfreeze_backbone(model, unfreeze_from_block=6):
    for i, layer in enumerate(model.features):
        if i >= unfreeze_from_block:
            for param in layer.parameters():
                param.requires_grad = True
    print(f"[Train] Unfroze backbone from block {unfreeze_from_block}+")


# ──────────────────────────────────────────────
# TRAIN / EVAL LOOPS
# ──────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                out  = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        correct      += (out.argmax(1) == labels).sum().item()
        total        += imgs.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        loss = criterion(out, labels)

        running_loss += loss.item() * imgs.size(0)
        preds = out.argmax(1)
        correct      += (preds == labels).sum().item()
        total        += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels


# ──────────────────────────────────────────────
# MAIN TRAINING LOOP
# ──────────────────────────────────────────────
def train(args):
    device      = get_device()
    num_workers = 0 if device.type == "mps" else 2

    train_loader, val_loader, found_classes = build_dataloaders(
        args.data_dir, args.batch_size, num_workers
    )
    num_classes = len(found_classes)
    print(f"\n[Model] Training for {num_classes} classes")

    model     = build_model(num_classes, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    warmup_epochs   = 3
    unfreeze_epoch  = warmup_epochs
    best_val_acc    = 0.0
    patience        = 5
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    ckpt_dir  = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_roadscan.pt"

    print(f"\n{'='*60}")
    print(f"  RoadScan AI — Training ({num_classes} Classes)")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch_size} | Device: {device}")
    print(f"  Image size: {IMAGE_SIZE}×{IMAGE_SIZE} | Classes from: dataset.yaml")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        if epoch == unfreeze_epoch + 1:
            unfreeze_backbone(model, unfreeze_from_block=6)
            optimizer = optim.AdamW([
                {"params": model.features.parameters(),    "lr": args.lr * 0.1},
                {"params": model.classifier.parameters(), "lr": args.lr},
            ], weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - epoch)
            print(f"[Phase 2] Backbone unfrozen — full fine-tuning begins")

        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        vl_loss, vl_acc, preds, labels_true = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.3f} | "
            f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.3f} | "
            f"{elapsed:.1f}s"
        )

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc     = vl_acc
            patience_counter = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_acc":     vl_acc,
                "classes":     found_classes,
                "class_to_idx": {c: i for i, c in enumerate(found_classes)},
                "image_size":  IMAGE_SIZE,
            }, ckpt_path)
            print(f"  💾 Saved best checkpoint (val_acc={vl_acc:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping (no improvement for {patience} epochs)")
            break

    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  ✅ TRAINING COMPLETE")
    print(f"  Best Val Acc : {best_val_acc:.4f}")
    print(f"  Checkpoint   : {ckpt_path}")
    print(f"  History      : {ckpt_dir / 'history.json'}")
    print(f"{'='*60}\n")

    try:
        from sklearn.metrics import classification_report
        print("\n📊 Per-Class Performance (final epoch):")
        print(classification_report(labels_true, preds, target_names=found_classes, digits=3))
    except ImportError:
        pass


# ──────────────────────────────────────────────
# INFERENCE HELPER
# ──────────────────────────────────────────────
@torch.no_grad()
def predict_image(image_path, ckpt_path):
    ckpt   = torch.load(ckpt_path, map_location="cpu")
    device = get_device()

    model = build_model(len(ckpt["classes"]), freeze_backbone=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, val_tf = get_transforms()
    img   = val_tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    probs = torch.softmax(model(img), dim=1)[0]
    idx   = probs.argmax().item()

    print(f"\n🎯 Prediction for: {image_path}")
    print(f"   Class      : {ckpt['classes'][idx]}")
    print(f"   Confidence : {probs[idx].item():.2%}")
    print(f"\n   Top-3 predictions:")
    top3 = torch.topk(probs, min(3, len(ckpt["classes"])))
    for i in range(top3.indices.numel()):
        cls = ckpt["classes"][top3.indices[i]]
        print(f"   {i+1}. {cls:30s} {top3.values[i].item():.2%}")

    return ckpt["classes"][idx], probs[idx].item()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="RoadScan AI — YAML + Prefix-Label Training")
    p.add_argument("--data_dir",     type=str,   default="./data",
                   help="Root containing dataset.yaml, train/images, val/images")
    p.add_argument("--output_dir",   type=str,   default="./checkpoints")
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--predict",      type=str,   default=None,
                   help="Path to an image for inference mode")
    p.add_argument("--ckpt",         type=str,   default=None,
                   help="Checkpoint path for --predict mode")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.predict:
        if not args.ckpt:
            raise ValueError("Provide --ckpt path/to/best_roadscan.pt with --predict")
        predict_image(args.predict, args.ckpt)
    else:
        data_dir = Path(args.data_dir)
        yaml_path = data_dir / "dataset.yaml"
        if not yaml_path.exists():
            print(f"\n❌ Error: dataset.yaml not found at {yaml_path}")
            print(f"\n💡 Expected layout:")
            print(f"   {args.data_dir}/")
            print(f"   ├── dataset.yaml    ← names, nc, train, val paths")
            print(f"   ├── train/images/   ← flat folder of images")
            print(f"   └── val/images/     ← flat folder of images")
            exit(1)

        train(args)