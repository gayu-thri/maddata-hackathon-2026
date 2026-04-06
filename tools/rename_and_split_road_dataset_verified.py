import re
from pathlib import Path
import shutil
import random

def rename_files(root_dir):
    root_path = Path(root_dir)
    
    for folder in root_path.iterdir():
        if not folder.is_dir():
            continue
        
        # Create prefix from folder name (remove spaces, underscores, hyphens)
        prefix = re.sub(r'[\s_\-]', '', folder.name).lower()
        
        for file in folder.iterdir():
            if not file.is_file():
                continue
            
            new_name = f"{prefix}_synthetic_{file.name}"
            new_path = file.parent / new_name
            
            file.rename(new_path)
            print(f"Renamed: {file.name} -> {new_name}")

def split_dataset(root_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    random.seed(seed)
    root_path = Path(root_dir)
    
    # Create split directories
    for split in ['train', 'val', 'test']:
        (root_path / split).mkdir(exist_ok=True)
    
    for folder in root_path.iterdir():
        if not folder.is_dir() or folder.name in ['train', 'val', 'test']:
            continue
        
        files = [f for f in folder.iterdir() if f.is_file()]
        random.shuffle(files)
        
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        splits = {
            'train': files[:n_train],
            'val':   files[n_train:n_train + n_val],
            'test':  files[n_train + n_val:]
        }
        
        for split_name, split_files in splits.items():
            dest_folder = root_path / split_name / folder.name
            dest_folder.mkdir(parents=True, exist_ok=True)
            
            for file in split_files:
                shutil.copy2(file, dest_folder / file.name)
                print(f"[{split_name}] {folder.name}/{file.name}")
        

if __name__ == "__main__":
    root_dir = "/Users/gayu/github/maddata-hackathon-2026/datasets/road_dataset_verified"
    # rename_files(root_dir)
    split_dataset(root_dir)
    print("Done!")