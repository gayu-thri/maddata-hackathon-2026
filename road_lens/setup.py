#!/usr/bin/env python3
"""
Quick Setup & Environment Check
Run this first to verify everything is installed correctly.
"""

import sys
import subprocess
from pathlib import Path

def check_installation():
    """Check if all required packages are installed."""
    
    print("\n" + "="*60)
    print("  🔍 RoadScan AI - Environment Check")
    print("="*60 + "\n")
    
    errors = []
    warnings = []
    
    # Check Python version
    print("📍 Python Version:")
    py_version = sys.version_info
    if py_version.major == 3 and py_version.minor >= 8:
        print(f"   ✅ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        print(f"   ❌ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
        errors.append("Python 3.8+ required")
    
    # Check PyTorch
    print("\n📍 PyTorch:")
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")
        
        # Check device availability
        if torch.cuda.is_available():
            print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            print("   ✅ Apple MPS available")
        else:
            print("   ⚠️  CPU only (training will be slower)")
            warnings.append("No GPU detected - consider using Colab")
    except ImportError:
        print("   ❌ PyTorch not installed")
        errors.append("PyTorch missing")
    
    # Check torchvision
    print("\n📍 Torchvision:")
    try:
        import torchvision
        print(f"   ✅ Torchvision {torchvision.__version__}")
    except ImportError:
        print("   ❌ Torchvision not installed")
        errors.append("Torchvision missing")
    
    # Check PIL
    print("\n📍 Pillow (PIL):")
    try:
        from PIL import Image
        import PIL
        print(f"   ✅ Pillow {PIL.__version__}")
    except ImportError:
        print("   ❌ Pillow not installed")
        errors.append("Pillow missing")
    
    # Check NumPy
    print("\n📍 NumPy:")
    try:
        import numpy as np
        print(f"   ✅ NumPy {np.__version__}")
    except ImportError:
        print("   ❌ NumPy not installed")
        errors.append("NumPy missing")
    
    # Check optional packages
    print("\n📍 Optional Packages:")
    try:
        import kaggle
        print("   ✅ Kaggle CLI installed")
        
        # Check Kaggle API token
        kaggle_path = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle_path.exists():
            print("   ✅ Kaggle API token found")
        else:
            print("   ⚠️  Kaggle API token not found")
            warnings.append("Setup Kaggle API for better datasets")
    except ImportError:
        print("   ⚠️  Kaggle CLI not installed")
        warnings.append("Install kaggle for better datasets: pip install kaggle --break-system-packages")
    
    try:
        import sklearn
        print("   ✅ scikit-learn installed")
    except ImportError:
        print("   ⚠️  scikit-learn not installed (optional)")
        warnings.append("Install scikit-learn for metrics: pip install scikit-learn --break-system-packages")
    
    # Summary
    print("\n" + "="*60)
    print("  📊 SUMMARY")
    print("="*60)
    
    if not errors:
        print("  ✅ All required packages installed!")
    else:
        print("  ❌ Missing required packages:")
        for err in errors:
            print(f"     • {err}")
        print("\n  💡 Install missing packages:")
        print("     pip3 install -r requirements.txt --break-system-packages")
    
    if warnings:
        print("\n  ⚠️  Warnings:")
        for warn in warnings:
            print(f"     • {warn}")
    
    print("="*60)
    
    return len(errors) == 0


def create_directories():
    """Create necessary directories."""
    
    print("\n📁 Creating directories...")
    
    dirs = [
        "./data/train",
        "./data/val",
        "./checkpoints",
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {d}")
    
    print("   ✅ Directories ready")


def print_next_steps():
    """Print next steps."""
    
    print("\n" + "="*60)
    print("  🚀 NEXT STEPS")
    print("="*60)
    print("\n  1. Collect Dataset (45 mins):")
    print("     python dataset_collector.py --output_dir ./data --images_per_class 500")
    print("\n  2. Train Model (60 mins):")
    print("     python roadscan_train_fast.py --data_dir ./data --epochs 20")
    print("\n  3. Test Model:")
    print("     python roadscan_train_fast.py --predict test.jpg --ckpt checkpoints/best_roadscan.pt")
    print("\n  💡 For detailed guide, see README.md")
    print("="*60 + "\n")


if __name__ == "__main__":
    success = check_installation()
    
    if success:
        create_directories()
        print_next_steps()
    else:
        print("\n❌ Please install missing packages first.")
        print("   Run: pip3 install -r requirements.txt --break-system-packages\n")
        sys.exit(1)
