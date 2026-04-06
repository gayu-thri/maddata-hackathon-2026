"""
RoadScan AI - Training Script
Fine-tunes EfficientNetV2-S (torchvision) for road hazard classification.

Designed to run on:
  - MacBook Pro M4 (MPS backend)
  - Google Colab (CUDA backend)
  - CPU fallback

Classes:
  0  broken_road_signs
  1  potholes
  2  littering
  3  damaged_roads
  4  illegal_parking
  5  road_debris
  6  faded_lane_markings
  7  flooded_roads
  8  other

Dataset directory layout expected:
  data/
    train/
      broken_road_signs/  *.jpg
      potholes/           *.jpg
      ...
    val/
      broken_road_signs/
      ...
    test/   (optional)
      ...

Usage:
  python roadscan_train.py --data_dir ./data --epochs 30 --batch_size 32
"""

import argparse
import os
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_V2_S_Weights
import numpy as np

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
CLASS_NAMES = [
    "broken_road_signs",
    "potholes",
    "littering",
    "damaged_roads",
    "illegal_parking",
    "road_debris",
    "faded_lane_markings",
    "flooded_roads",
    "other",
]
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE  = 384   # EfficientNetV2-S native resolution


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
        print("[Device] CPU (consider Colab for faster training)")
    return device


# ──────────────────────────────────────────────
# DATA TRANSFORMS
# ──────────────────────────────────────────────
def get_transforms():
    # ImageNet stats — EfficientNetV2 was pretrained on these
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.RandomRotation(15),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        # Simulate dashcam / phone blur / noise
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, val_tf


# ──────────────────────────────────────────────
# DATASET + WEIGHTED SAMPLER (handles class imbalance)
# ──────────────────────────────────────────────
def build_dataloaders(data_dir, batch_size, num_workers):
    train_tf, val_tf = get_transforms()
    data_dir = Path(data_dir)

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(data_dir / "val",   transform=val_tf)

    # Weighted sampler so rare classes get fair representation
    targets = torch.tensor(train_ds.targets)
    class_count = torch.bincount(targets).float()
    class_count[class_count == 0] = 1
    weights = 1.0 / class_count[targets]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )

    print(f"[Data] Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"[Data] Classes found: {train_ds.classes}")
    return train_loader, val_loader, train_ds.classes


# ──────────────────────────────────────────────
# MODEL — EfficientNetV2-S
# Alternatives also available: convnext_base, vit_b_16
# ──────────────────────────────────────────────
def build_model(num_classes, freeze_backbone=True):
    """
    EfficientNetV2-S: great accuracy/speed tradeoff, well-suited for
    fine-grained visual classification on limited GPU/MPS budgets.
    """
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model   = models.efficientnet_v2_s(weights=weights)

    if freeze_backbone:
        # Phase 1: only train the classifier head
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )
    return model


def unfreeze_backbone(model, unfreeze_from_block=5):
    """Gradually unfreeze deeper blocks for phase-2 fine-tuning."""
    for i, layer in enumerate(model.features):
        if i >= unfreeze_from_block:
            for param in layer.parameters():
                param.requires_grad = True
    print(f"[Train] Unfroze backbone from block {unfreeze_from_block}+")


# ──────────────────────────────────────────────
# LABEL SMOOTHING LOSS
# ──────────────────────────────────────────────
class LabelSmoothingCE(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.cls = classes

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_val  = self.smoothing / (self.cls - 1)
        one_hot = torch.full_like(pred, smooth_val)
        one_hot.scatter_(1, target.unsqueeze(1), confidence)
        log_prob = nn.functional.log_softmax(pred, dim=1)
        return -(one_hot * log_prob).sum(dim=1).mean()


# ──────────────────────────────────────────────
# TRAIN / EVAL LOOPS
# ──────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # AMP for CUDA; MPS uses float32 autocast (skip if unavailable)
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                out  = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        preds     = out.argmax(1)
        correct      += (preds == labels).sum().item()
        total        += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels


# ──────────────────────────────────────────────
# MAIN TRAINING LOOP
# ──────────────────────────────────────────────
def train(args):
    device  = get_device()
    num_workers = 0 if device.type == "mps" else 4  # MPS prefers 0 workers

    train_loader, val_loader, found_classes = build_dataloaders(
        args.data_dir, args.batch_size, num_workers
    )
    num_classes = len(found_classes)
    print(f"[Model] Training for {num_classes} classes")

    model = build_model(num_classes, freeze_backbone=True).to(device)
    criterion = LabelSmoothingCE(num_classes, smoothing=0.1)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # ── Phase 1: warm-up head only ──
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_head, weight_decay=1e-4
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    warmup_epochs = min(5, args.epochs // 4)
    unfreeze_epoch = warmup_epochs

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    ckpt_path = Path(args.output_dir) / "best_roadscan.pt"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  RoadScan AI — EfficientNetV2-S Fine-Tuning")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch_size} | Device: {device}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Phase 2: unfreeze backbone after warm-up
        if epoch == unfreeze_epoch + 1:
            unfreeze_backbone(model, unfreeze_from_block=5)
            optimizer = optim.AdamW([
                {"params": model.features.parameters(), "lr": args.lr_backbone},
                {"params": model.classifier.parameters(), "lr": args.lr_head},
            ], weight_decay=1e-4)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
            print(f"[Phase 2] Backbone unfrozen — full fine-tuning started")

        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        vl_loss, vl_acc, preds, labels_true = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | "
            f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.3f} | "
            f"{elapsed:.1f}s"
        )

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_acc":     vl_acc,
                "classes":     found_classes,
                "class_names": CLASS_NAMES,
                "image_size":  IMAGE_SIZE,
            }, ckpt_path)
            print(f"Saved best checkpoint (val_acc={vl_acc:.4f})")

    # Save training history
    with open(Path(args.output_dir) / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[Done] Best Val Acc: {best_val_acc:.4f}")
    print(f"[Done] Checkpoint: {ckpt_path}")
    print(f"[Done] History: {Path(args.output_dir) / 'history.json'}")


# ──────────────────────────────────────────────
# INFERENCE HELPER (quick sanity check)
# ──────────────────────────────────────────────
@torch.no_grad()
def predict_image(image_path, ckpt_path):
    """
    Load checkpoint and predict a single image.
    Returns (class_name, confidence)
    """
    from PIL import Image

    ckpt   = torch.load(ckpt_path, map_location="cpu")
    device = get_device()

    model = build_model(len(ckpt["classes"]), freeze_backbone=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, val_tf = get_transforms()
    img   = val_tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    probs = torch.softmax(model(img), dim=1)[0]
    idx   = probs.argmax().item()
    return ckpt["classes"][idx], probs[idx].item()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="RoadScan AI — Training Script")
    p.add_argument("--data_dir",    type=str, default="./data",
                   help="Root of train/val ImageFolder dataset")
    p.add_argument("--output_dir",  type=str, default="./checkpoints",
                   help="Directory to save checkpoints and logs")
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--batch_size",  type=int, default=32,
                   help="Reduce to 16 on M4 / Colab T4 if OOM")
    p.add_argument("--lr_head",     type=float, default=1e-3,
                   help="LR for classifier head (phase 1 + 2)")
    p.add_argument("--lr_backbone", type=float, default=2e-5,
                   help="LR for unfrozen backbone layers (phase 2)")
    p.add_argument("--predict",     type=str, default=None,
                   help="Path to image for inference (skips training)")
    p.add_argument("--ckpt",        type=str, default=None,
                   help="Checkpoint path for --predict mode")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.predict:
        if not args.ckpt:
            raise ValueError("Pass --ckpt path/to/best_roadscan.pt with --predict")
        cls, conf = predict_image(args.predict, args.ckpt)
        print(f"Prediction: {cls}  (confidence: {conf:.2%})")
    else:
        train(args)