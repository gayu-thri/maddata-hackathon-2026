"""
RDD2022 Dataset Downloader & Converter
Downloads Road Damage Detection dataset and converts bounding boxes to cropped classification images.

RDD2022 has classes like:
  - D00: Longitudinal Crack
  - D10: Transverse Crack  
  - D20: Alligator Crack
  - D40: Pothole

This script:
  1. Downloads RDD2022 subset from Roboflow (or provides instructions)
  2. Converts YOLO/COCO format → cropped images
  3. Maps to our classes (potholes, cracked_pavement)

Usage:
  python rdd_converter.py --rdd_dir ./rdd2022 --output_dir ./data
"""

import argparse
import json
import shutil
from pathlib import Path
from PIL import Image
import random


# RDD2022 class mapping to our categories
RDD_CLASS_MAPPING = {
    # RDD2022 class → Our class
    "D40": "potholes",          # Pothole
    "D43": "potholes",          # White line blur (pothole related)
    "D44": "potholes",          # Transverse rumbling (pothole related)
    
    "D00": "cracked_pavement",  # Longitudinal Crack
    "D10": "cracked_pavement",  # Transverse Crack
    "D20": "cracked_pavement",  # Alligator Crack
}


def convert_yolo_to_bbox(yolo_line, img_width, img_height):
    """
    Convert YOLO format to absolute bounding box coordinates.
    YOLO format: class_id x_center y_center width height (all normalized 0-1)
    Returns: (x_min, y_min, x_max, y_max, class_id)
    """
    parts = yolo_line.strip().split()
    class_id = int(parts[0])
    x_center = float(parts[1]) * img_width
    y_center = float(parts[2]) * img_height
    width = float(parts[3]) * img_width
    height = float(parts[4]) * img_height
    
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)
    
    return (x_min, y_min, x_max, y_max, class_id)


def crop_and_save_boxes(image_path, label_path, class_names, output_dir, val_split=0.15):
    """
    Crop bounding boxes from an image and save to appropriate class folders.
    """
    if not label_path.exists():
        return 0
    
    try:
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        crop_count = 0
        
        with open(label_path, 'r') as f:
            for line in f:
                x_min, y_min, x_max, y_max, class_id = convert_yolo_to_bbox(
                    line, img_width, img_height
                )
                
                # Get class name
                if class_id >= len(class_names):
                    continue
                    
                rdd_class = class_names[class_id]
                
                # Map to our classes
                if rdd_class not in RDD_CLASS_MAPPING:
                    continue
                    
                our_class = RDD_CLASS_MAPPING[rdd_class]
                
                # Decide train or val
                split = "val" if random.random() < val_split else "train"
                
                # Create output directory
                class_dir = output_dir / split / our_class
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Crop and save
                try:
                    # Add small padding
                    padding = 10
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(img_width, x_max + padding)
                    y_max = min(img_height, y_max + padding)
                    
                    cropped = img.crop((x_min, y_min, x_max, y_max))
                    
                    # Skip very small crops
                    if cropped.width < 50 or cropped.height < 50:
                        continue
                    
                    # Save with unique name
                    img_name = image_path.stem
                    save_name = f"rdd_{img_name}_{crop_count:03d}.jpg"
                    save_path = class_dir / save_name
                    
                    cropped.save(save_path, quality=95)
                    crop_count += 1
                    
                except Exception as e:
                    print(f"  ⚠️  Failed to crop box from {image_path.name}: {e}")
                    continue
        
        return crop_count
        
    except Exception as e:
        print(f"  ⚠️  Failed to process {image_path.name}: {e}")
        return 0


def convert_rdd_dataset(rdd_dir, output_dir):
    """
    Convert RDD2022 YOLO format dataset to our classification format.
    
    Expected RDD directory structure:
      rdd2022/
        images/
          train/
            *.jpg
          val/
            *.jpg
        labels/
          train/
            *.txt
          val/
            *.txt
        data.yaml  (contains class names)
    """
    rdd_dir = Path(rdd_dir)
    output_dir = Path(output_dir)
    
    print("\n" + "="*60)
    print("  🔄 Converting RDD2022 to Classification Format")
    print("="*60)
    
    # Read class names from data.yaml
    data_yaml = rdd_dir / "data.yaml"
    class_names = ["D00", "D10", "D20", "D40", "D43", "D44"]  # Default RDD classes
    
    if data_yaml.exists():
        try:
            import yaml
            with open(data_yaml, 'r') as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    class_names = data['names']
        except:
            print("  ℹ️  Could not read data.yaml, using default class names")
    
    print(f"  📋 RDD Classes: {class_names}")
    print(f"  🎯 Mapping to: {set(RDD_CLASS_MAPPING.values())}")
    
    total_crops = 0
    
    # Process train and val splits
    for split in ['train', 'valid']:  # RDD uses 'valid' instead of 'val'
        images_dir = rdd_dir / 'images' / split
        labels_dir = rdd_dir / 'labels' / split
        
        if not images_dir.exists():
            print(f"  ⚠️  {images_dir} not found, skipping...")
            continue
        
        print(f"\n  📂 Processing {split} split...")
        
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        print(f"  📊 Found {len(image_files)} images")
        
        for i, img_path in enumerate(image_files):
            label_path = labels_dir / f"{img_path.stem}.txt"
            crops = crop_and_save_boxes(img_path, label_path, class_names, output_dir)
            total_crops += crops
            
            if (i + 1) % 50 == 0:
                print(f"    Processed {i+1}/{len(image_files)} images ({total_crops} crops)", end='\r')
        
        print(f"    ✓ Processed {len(image_files)} images ({total_crops} crops)")
    
    # Summary
    print("\n" + "="*60)
    print("  📊 CONVERSION SUMMARY")
    print("="*60)
    
    for our_class in set(RDD_CLASS_MAPPING.values()):
        train_dir = output_dir / "train" / our_class
        val_dir = output_dir / "val" / our_class
        
        train_count = len(list(train_dir.glob('*'))) if train_dir.exists() else 0
        val_count = len(list(val_dir.glob('*'))) if val_dir.exists() else 0
        
        print(f"  ✓ {our_class:20s} | Train: {train_count:4d} | Val: {val_count:4d}")
    
    print("="*60)
    print(f"  Total cropped images: {total_crops}")
    print("="*60)
    
    return total_crops


def download_instructions():
    """Print instructions for downloading RDD2022."""
    print("\n" + "="*60)
    print("  📥 HOW TO GET RDD2022 DATASET")
    print("="*60)
    
    print("\n  Option 1: Roboflow (Easiest)")
    print("  ────────────────────────────")
    print("  1. Go to: https://universe.roboflow.com/sekilab/rdd2022")
    print("  2. Click 'Download Dataset'")
    print("  3. Select format: YOLOv5 PyTorch")
    print("  4. Download and extract to a folder (e.g., ./rdd2022)")
    print("  5. Run: python rdd_converter.py --rdd_dir ./rdd2022 --output_dir ./data")
    
    print("\n  Option 2: Official RDD2022")
    print("  ──────────────────────────")
    print("  1. Visit: https://github.com/sekilab/RoadDamageDetector")
    print("  2. Download the dataset from their releases")
    print("  3. Convert to YOLO format if needed")
    
    print("\n  Option 3: Direct Kaggle")
    print("  ───────────────────────")
    print("  Search Kaggle for 'RDD2022' or 'Road Damage Detection'")
    
    print("\n" + "="*60)


def parse_args():
    parser = argparse.ArgumentParser(description="RDD2022 Converter")
    parser.add_argument(
        "--rdd_dir",
        type=str,
        help="Path to RDD2022 dataset directory (YOLO format)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory for converted dataset"
    )
    parser.add_argument(
        "--show_instructions",
        action="store_true",
        help="Show download instructions and exit"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.show_instructions or not args.rdd_dir:
        download_instructions()
        if not args.rdd_dir:
            print("\n💡 Run with --rdd_dir <path> after downloading the dataset")
            exit(0)
    
    if not Path(args.rdd_dir).exists():
        print(f"\n❌ Error: {args.rdd_dir} does not exist")
        print("\n💡 Run with --show_instructions to see download options")
        exit(1)
    
    try:
        import yaml
    except ImportError:
        print("\n⚠️  PyYAML not found. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'pyyaml', '--break-system-packages'])
        import yaml
    
    crops = convert_rdd_dataset(args.rdd_dir, args.output_dir)
    
    if crops > 0:
        print(f"\n✅ Successfully extracted {crops} images from RDD2022!")
        print(f"📁 Dataset ready at: {args.output_dir}")
    else:
        print("\n⚠️  No images were extracted. Check your RDD directory structure.")
        