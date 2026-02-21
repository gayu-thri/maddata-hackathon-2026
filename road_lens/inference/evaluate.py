#!/usr/bin/env python3
"""
RoadScan AI - Batch Testing & Demo
Test your trained model on multiple images and generate a report.

Usage:
  python demo_test.py --ckpt checkpoints/best_roadscan.pt --test_dir ./test_images
  python demo_test.py --ckpt checkpoints/best_roadscan.pt --image single_test.jpg
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import EfficientNet_V2_S_Weights
from PIL import Image
import numpy as np


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes):
    """Rebuild model architecture (must match training)."""
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = models.efficientnet_v2_s(weights=weights)
    
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(256, num_classes),
    )
    return model


def load_model(ckpt_path):
    """Load trained model from checkpoint."""
    print(f"📦 Loading model from: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    device = get_device()
    
    num_classes = len(ckpt["classes"])
    model = build_model(num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    print(f"   ✅ Model loaded (trained for {num_classes} classes)")
    print(f"   📊 Validation accuracy: {ckpt['val_acc']:.2%}")
    print(f"   🏷️  Classes: {', '.join(ckpt['classes'])}")
    
    return model, ckpt, device


def get_transform(image_size=224):
    """Get validation transform (same as training)."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


@torch.no_grad()
def predict_image(image_path, model, transform, class_names, device):
    """Predict a single image."""
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu()
        
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()
        
        # Get top 3
        top3_probs, top3_indices = torch.topk(probs, min(3, len(class_names)))
        top3 = [(class_names[idx], prob.item()) for idx, prob in zip(top3_indices, top3_probs)]
        
        return {
            "prediction": class_names[pred_idx],
            "confidence": confidence,
            "top3": top3,
            "success": True,
        }
    except Exception as e:
        return {
            "prediction": None,
            "confidence": 0.0,
            "top3": [],
            "success": False,
            "error": str(e),
        }


def test_single_image(image_path, ckpt_path):
    """Test a single image and print results."""
    model, ckpt, device = load_model(ckpt_path)
    transform = get_transform(ckpt.get("image_size", 224))
    
    print(f"\n{'='*60}")
    print(f"  🖼️  Testing Image: {image_path}")
    print(f"{'='*60}\n")
    
    result = predict_image(image_path, model, transform, ckpt["classes"], device)
    
    if result["success"]:
        print(f"🎯 Prediction: {result['prediction']}")
        print(f"📊 Confidence: {result['confidence']:.2%}")
        print(f"\n🏆 Top 3 Predictions:")
        for i, (cls, prob) in enumerate(result["top3"], 1):
            bar = "█" * int(prob * 40)
            print(f"   {i}. {cls:20s} {prob:6.2%} {bar}")
    else:
        print(f"❌ Error: {result['error']}")
    
    print(f"\n{'='*60}\n")


def test_directory(test_dir, ckpt_path, output_file=None):
    """Test all images in a directory."""
    model, ckpt, device = load_model(ckpt_path)
    transform = get_transform(ckpt.get("image_size", 224))
    
    test_dir = Path(test_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [f for f in test_dir.rglob("*") if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"❌ No images found in {test_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"  📁 Testing Directory: {test_dir}")
    print(f"  📊 Found {len(image_files)} images")
    print(f"{'='*60}\n")
    
    results = []
    class_counts = {cls: 0 for cls in ckpt["classes"]}
    confidence_sum = 0.0
    
    for i, img_path in enumerate(image_files, 1):
        result = predict_image(img_path, model, transform, ckpt["classes"], device)
        
        if result["success"]:
            results.append({
                "file": str(img_path.name),
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "top3": result["top3"],
            })
            class_counts[result["prediction"]] += 1
            confidence_sum += result["confidence"]
            
            print(f"{i:3d}/{len(image_files)} | {img_path.name:30s} → {result['prediction']:20s} ({result['confidence']:.1%})")
        else:
            print(f"{i:3d}/{len(image_files)} | {img_path.name:30s} → ERROR: {result['error']}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  📊 SUMMARY")
    print(f"{'='*60}")
    print(f"  Total images: {len(image_files)}")
    print(f"  Successful: {len(results)}")
    print(f"  Avg confidence: {confidence_sum / len(results):.1%}" if results else "  Avg confidence: N/A")
    
    print(f"\n  📈 Predictions by Class:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = count / len(results) * 100 if results else 0
            bar = "█" * int(pct / 2)
            print(f"     {cls:20s}: {count:3d} ({pct:5.1f}%) {bar}")
    
    print(f"{'='*60}\n")
    
    # Save results
    if output_file:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "model": str(ckpt_path),
            "test_dir": str(test_dir),
            "total_images": len(image_files),
            "successful": len(results),
            "avg_confidence": confidence_sum / len(results) if results else 0.0,
            "class_counts": class_counts,
            "results": results,
        }
        
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Results saved to: {output_file}\n")


def main():
    parser = argparse.ArgumentParser(description="RoadScan AI - Testing & Demo")
    parser.add_argument("--ckpt", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--image", type=str, default=None,
                       help="Test a single image")
    parser.add_argument("--test_dir", type=str, default=None,
                       help="Test all images in a directory")
    parser.add_argument("--output", type=str, default=None,
                       help="Save results to JSON file (for --test_dir)")
    
    args = parser.parse_args()
    
    if not Path(args.ckpt).exists():
        print(f"❌ Error: Checkpoint not found: {args.ckpt}")
        return
    
    if args.image:
        if not Path(args.image).exists():
            print(f"❌ Error: Image not found: {args.image}")
            return
        test_single_image(args.image, args.ckpt)
    
    elif args.test_dir:
        if not Path(args.test_dir).exists():
            print(f"❌ Error: Directory not found: {args.test_dir}")
            return
        test_directory(args.test_dir, args.ckpt, args.output)
    
    else:
        print("❌ Error: Specify either --image or --test_dir")
        print("   Example: python demo_test.py --ckpt checkpoints/best_roadscan.pt --image test.jpg")


if __name__ == "__main__":
    main()