"""
RoadScan AI - Quick Training Script
Fine-tunes EfficientNetV2-S for 6-class road hazard classification.

Classes:
  0  potholes
  1  cracked_pavement
  2  road_debris
  3  normal_road
  4  broken_road_signs
  5  faded_lane_markings

Optimizations vs original:
  - Reduced epochs (20 vs 30)
  - Smaller batch size (16 vs 32) for M4 Mac
  - Faster warm-up (3 epochs vs 5)
  - Simplified augmentations
  - Early stopping

Usage:
  python roadscan_train_fast.py --data_dir ./data --epochs 20 --batch_size 16
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_V2_S_Weights
import numpy as np

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
CLASS_NAMES = [
    "potholes",
    "cracked_pavement",
    "road_debris",
    "normal_road",
    "broken_road_signs",
    "faded_lane_markings",
]
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE  = 224   # Smaller than 384 for faster training


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
# FASTER DATA TRANSFORMS (simplified augmentation)
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

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(data_dir / "val",   transform=val_tf)

    # Weighted sampler for class imbalance
    targets = torch.tensor(train_ds.targets)
    class_count = torch.bincount(targets).float()
    class_count[class_count == 0] = 1
    weights = 1.0 / class_count[targets]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, 
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    print(f"[Data] Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"[Data] Classes: {train_ds.classes}")
    
    # Show class distribution
    print(f"[Data] Class distribution:")
    for i, cls in enumerate(train_ds.classes):
        count = (targets == i).sum().item()
        print(f"       {cls:20s}: {count:4d} images")
    
    return train_loader, val_loader, train_ds.classes


# ──────────────────────────────────────────────
# MODEL — EfficientNetV2-S
# ──────────────────────────────────────────────
def build_model(num_classes, freeze_backbone=True):
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model   = models.efficientnet_v2_s(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Simpler classifier for faster training
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
    """Unfreeze last few blocks only for faster fine-tuning."""
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
        preds     = out.argmax(1)
        correct      += (preds == labels).sum().item()
        total        += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels


# ──────────────────────────────────────────────
# MAIN TRAINING LOOP (OPTIMIZED)
# ──────────────────────────────────────────────
def train(args):
    device  = get_device()
    num_workers = 0 if device.type == "mps" else 2  # Reduced workers for speed

    train_loader, val_loader, found_classes = build_dataloaders(
        args.data_dir, args.batch_size, num_workers
    )
    num_classes = len(found_classes)
    print(f"[Model] Training for {num_classes} classes")

    model = build_model(num_classes, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()  # No label smoothing for speed
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Phase 1: warm-up head only
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    
    # Simple cosine schedule
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    warmup_epochs = 3  # Quick warm-up
    unfreeze_epoch = warmup_epochs

    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    ckpt_path = Path(args.output_dir) / "best_roadscan.pt"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  RoadScan AI — Fast Training (6 Classes)")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch_size} | Device: {device}")
    print(f"  Image Size: {IMAGE_SIZE}x{IMAGE_SIZE} (optimized for speed)")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Phase 2: unfreeze backbone after warm-up
        if epoch == unfreeze_epoch + 1:
            unfreeze_backbone(model, unfreeze_from_block=6)
            optimizer = optim.AdamW([
                {"params": model.features.parameters(), "lr": args.lr * 0.1},
                {"params": model.classifier.parameters(), "lr": args.lr},
            ], weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - epoch)
            print(f"[Phase 2] Backbone unfrozen — full fine-tuning")

        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        vl_loss, vl_acc, preds, labels_true = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | "
            f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.3f} | "
            f"{elapsed:.1f}s"
        )

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        # Save best model
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            patience_counter = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_acc":     vl_acc,
                "classes":     found_classes,
                "class_names": CLASS_NAMES,
                "image_size":  IMAGE_SIZE,
            }, ckpt_path)
            print(f"💾 Saved best checkpoint (val_acc={vl_acc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping triggered (no improvement for {patience} epochs)")
            break

    # Save training history
    with open(Path(args.output_dir) / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  ✅ TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best Val Acc: {best_val_acc:.4f}")
    print(f"  Checkpoint:   {ckpt_path}")
    print(f"  History:      {Path(args.output_dir) / 'history.json'}")
    print(f"{'='*60}\n")
    
    # Print per-class accuracy if possible
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        print("\n📊 Per-Class Performance (final epoch):")
        print(classification_report(labels_true, preds, target_names=found_classes, digits=3))
    except:
        pass


# ──────────────────────────────────────────────
# INFERENCE HELPER
# ──────────────────────────────────────────────
@torch.no_grad()
def predict_image(image_path, ckpt_path):
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
    
    print(f"\n🎯 Prediction for: {image_path}")
    print(f"   Class: {ckpt['classes'][idx]}")
    print(f"   Confidence: {probs[idx].item():.2%}")
    print(f"\n   Top 3 predictions:")
    top3 = torch.topk(probs, 3)
    for i in range(3):
        print(f"   {i+1}. {ckpt['classes'][top3.indices[i]]:20s} {top3.values[i].item():.2%}")
    
    return ckpt["classes"][idx], probs[idx].item()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="RoadScan AI — Fast Training")
    p.add_argument("--data_dir",    type=str, default="./data",
                   help="Root of train/val ImageFolder dataset")
    p.add_argument("--output_dir",  type=str, default="./checkpoints",
                   help="Directory to save checkpoints")
    p.add_argument("--epochs",      type=int, default=20,
                   help="Reduced for 4-hour timeline")
    p.add_argument("--batch_size",  type=int, default=16,
                   help="Reduced for M4 Mac")
    p.add_argument("--lr",          type=float, default=1e-3,
                   help="Learning rate")
    p.add_argument("--predict",     type=str, default=None,
                   help="Path to image for inference")
    p.add_argument("--ckpt",        type=str, default=None,
                   help="Checkpoint path for --predict mode")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.predict:
        if not args.ckpt:
            raise ValueError("Pass --ckpt path/to/best_roadscan.pt with --predict")
        predict_image(args.predict, args.ckpt)
    else:
        # Check if dataset exists
        data_dir = Path(args.data_dir)
        if not (data_dir / "train").exists():
            print(f"\n❌ Error: {data_dir / 'train'} not found")
            print("\n💡 Run dataset_collector.py first to download images:")
            print(f"   python dataset_collector.py --output_dir {args.data_dir}")
            exit(1)
        
        train(args)
