#!/usr/bin/env python3
"""
Finalize dataset into a valid YOLOv8 layout.
Fixed: Handles flat images/labels structure from merge_and_upload_datasets.py
"""

from pathlib import Path
import shutil
import random
import yaml

INPUT_DIR = Path("/workspace/dataset_yolo")
OUTPUT_DIR = Path("/workspace/dataset_yolo")

CLASS_NAMES = ["Bear", "Boar", "Deer", "Elephant", "Tapir", "Tiger"]
SPLITS = ["train", "val", "test"]

def main():
    print(f"[1/4] Checking dataset structure...")
    
    # Check if we have the flat structure (images/train/*.jpg)
    if (INPUT_DIR / "images" / "train").exists():
        print("[INFO] Found structure: images/train/ with flat files")
        
        # Check if we need to reorganize
        images_count = 0
        labels_count = 0
        
        for split in SPLITS:
            img_dir = INPUT_DIR / "images" / split
            lbl_dir = INPUT_DIR / "labels" / split
            
            if img_dir.exists():
                imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
                images_count += len(imgs)
                print(f"  {split}: {len(imgs)} images")
            
            if lbl_dir.exists():
                lbls = list(lbl_dir.glob("*.txt"))
                labels_count += len(lbls)
        
        print(f"[OK] Found {images_count} images, {labels_count} labels")
        
        if images_count == 0:
            print("[ERROR] No images found")
            return 1
        
        # Create YOLOv8 structure (train/images/, train/labels/)
        print("\n[2/4] Creating YOLOv8 structure...")
        
        for split in SPLITS:
            # Create new structure
            new_img_dir = INPUT_DIR / split / "images"
            new_lbl_dir = INPUT_DIR / split / "labels"
            new_img_dir.mkdir(parents=True, exist_ok=True)
            new_lbl_dir.mkdir(parents=True, exist_ok=True)
            
            # Move files from old structure
            old_img_dir = INPUT_DIR / "images" / split
            old_lbl_dir = INPUT_DIR / "labels" / split
            
            if old_img_dir.exists():
                for img in old_img_dir.glob("*"):
                    if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        shutil.move(str(img), str(new_img_dir / img.name))
                
                # Remove old directory if empty
                try:
                    old_img_dir.rmdir()
                except:
                    pass
            
            if old_lbl_dir.exists():
                for lbl in old_lbl_dir.glob("*.txt"):
                    shutil.move(str(lbl), str(new_lbl_dir / lbl.name))
                
                try:
                    old_lbl_dir.rmdir()
                except:
                    pass
            
            print(f"  [OK] {split} restructured")
        
        # Clean up old top-level directories
        try:
            (INPUT_DIR / "images").rmdir()
        except:
            pass
        try:
            (INPUT_DIR / "labels").rmdir()
        except:
            pass
    
    else:
        print("[INFO] Structure already correct or no images found")
    
    # Create/Update data.yaml
    print("\n[3/4] Creating data.yaml...")
    
    yaml_path = OUTPUT_DIR / "data.yaml"
    data = {
        "path": str(OUTPUT_DIR.absolute()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images" if (OUTPUT_DIR / "test").exists() else "val/images",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
        "nc": len(CLASS_NAMES)
    }
    
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)
    
    print(f"[OK] data.yaml written to {yaml_path}")
    
    # Final validation
    print("\n[4/4] Validating final structure...")
    
    total_images = 0
    for split in SPLITS:
        img_dir = OUTPUT_DIR / split / "images"
        if img_dir.exists():
            count = len(list(img_dir.glob("*")))
            total_images += count
            print(f"  {split}/images/: {count} images")
    
    if total_images == 0:
        print("[ERROR] No images in final dataset")
        return 1
    
    print(f"\n[SUCCESS] Dataset finalized: {OUTPUT_DIR}")
    print(f"  Total images: {total_images}")
    print(f"  Classes: {len(CLASS_NAMES)}")
    
    return 0

if __name__ == "__main__":
    exit(main())
    