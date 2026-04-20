#!/usr/bin/env python3
"""
Smart Dataset Merger v3 - Pulls from MinIO, then processes
"""

from pathlib import Path
import shutil
import random
import yaml
import os
from collections import defaultdict
from typing import Optional
from minio import Minio
import tarfile
import zipfile

STAGING_DIR = Path("/workspace/datasets_staging")
OUTPUT_DIR = Path("/workspace/dataset_yolo")

# MinIO configuration
MINIO_URL = os.environ.get("MINIO_URL", "minio:9000")
MINIO_USER = os.environ.get("MINIO_USER", "minioadmin")
MINIO_PASSWORD = os.environ.get("MINIO_PASSWORD", "minioadmin")
MINIO_BUCKET = "datasets"

# Standard class mapping
CLASS_MAP = {
    "Bear": 0, "Boar": 1, "Deer": 2, 
    "Elephant": 3, "Tapir": 4, "Tiger": 5
}

# Map Romanian/Roboflow classes to standard classes
CLASS_ALIASES = {
    # Bears
    "Urs": "Bear", "SunBear": "Bear", "brown bear": "Bear", "black bear": "Bear",
    # Boars
    "Mistret": "Boar", "wild boar": "Boar", "pig": "Boar",
    # Deer
    "Cerb Comun": "Deer", "Cerb": "Deer", "elk": "Deer", "deer": "Deer",
}

# Reverse map for YOLO class IDs
YOLO_CLASS_NAMES = ["Bear", "Boar", "Deer", "Elephant", "Tapir", "Tiger"]


def download_from_minio():
    """Download all datasets from MinIO to staging directory"""
    print("="*60)
    print("DOWNLOADING FROM MINIO")
    print("="*60)
    
    client = Minio(
        MINIO_URL,
        access_key=MINIO_USER,
        secret_key=MINIO_PASSWORD,
        secure=False
    )
    
    if not client.bucket_exists(MINIO_BUCKET):
        print(f"[WARN] Bucket '{MINIO_BUCKET}' does not exist")
        return False
    
    # Clear staging directory
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    
    # List and download all objects
    objects = list(client.list_objects(MINIO_BUCKET, recursive=True))
    print(f"[INFO] Found {len(objects)} objects in MinIO bucket '{MINIO_BUCKET}'")
    
    downloaded = 0
    for obj in objects:
        # Skip directories
        if obj.object_name.endswith('/'):
            continue
        
        # Create local path
        local_path = STAGING_DIR / obj.object_name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            client.fget_object(MINIO_BUCKET, obj.object_name, str(local_path))
            downloaded += 1
            
            # Extract if it's an archive
            if local_path.suffix in ['.zip', '.tar', '.gz']:
                extract_archive(local_path)
                
        except Exception as e:
            print(f"[ERROR] Failed to download {obj.object_name}: {e}")
    
    print(f"[OK] Downloaded {downloaded} files from MinIO")
    return True


def extract_archive(archive_path: Path):
    """Extract zip/tar archives"""
    extract_dir = archive_path.parent / archive_path.stem
    
    try:
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(extract_dir)
            print(f"  [EXTRACT] {archive_path.name} -> {extract_dir}")
        elif archive_path.suffix in ['.tar', '.gz'] or '.tar' in archive_path.name:
            with tarfile.open(archive_path, 'r') as tf:
                tf.extractall(extract_dir)
            print(f"  [EXTRACT] {archive_path.name} -> {extract_dir}")
    except Exception as e:
        print(f"  [WARN] Could not extract {archive_path}: {e}")


def normalize_class(name: str) -> Optional[str]:
    """Convert any class name to standard class"""
    if not name:
        return None
    
    name = str(name).strip()
    
    # Check aliases first
    for alias, target in CLASS_ALIASES.items():
        if alias.lower() == name.lower():
            return target
    
    # Check direct match
    if name in CLASS_MAP:
        return name
    
    return None


def process_yolo_dataset(dataset_path: Path) -> list:
    """Process dataset already in YOLO format"""
    images = []
    
    for split in ["train", "val", "valid", "test"]:
        img_dir = dataset_path / split / "images"
        lbl_dir = dataset_path / split / "labels"
        
        if not img_dir.exists():
            continue
        
        target_split = "val" if split == "valid" else split
        
        print(f"  Scanning {split}/images...")
        
        for img_path in img_dir.glob("*"):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            
            # Find corresponding label
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            
            # If label exists, read class
            class_name = None
            if lbl_path.exists():
                try:
                    with open(lbl_path) as f:
                        first_line = f.readline().strip()
                        if first_line:
                            parts = first_line.split()
                            if parts:
                                class_id = int(parts[0])
                                if class_id < len(YOLO_CLASS_NAMES):
                                    class_name = YOLO_CLASS_NAMES[class_id]
                except Exception:
                    pass
            
            # If no label or couldn't read, infer from path
            if not class_name:
                class_name = infer_class_from_path(img_path)
            
            if class_name:
                images.append((img_path, lbl_path if lbl_path.exists() else None, 
                              class_name, target_split))
    
    return images


def process_classification_dataset(dataset_path: Path) -> list:
    """Process dataset in classification format (class subfolders)"""
    images = []
    
    for split in ["train", "valid", "val", "test"]:
        split_dir = dataset_path / split
        
        if not split_dir.exists():
            continue
        
        target_split = "val" if split == "valid" else split
        print(f"  Scanning {split}/...")
        
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = normalize_class(class_dir.name)
            if not class_name:
                print(f"    [SKIP] Unknown class: {class_dir.name}")
                continue
            
            img_count = 0
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    images.append((img_path, None, class_name, target_split))
                    img_count += 1
            
            if img_count > 0:
                print(f"    {class_dir.name} -> {class_name}: {img_count} images")
    
    return images


def infer_class_from_path(img_path: Path) -> Optional[str]:
    """Try to infer class from path"""
    for parent in img_path.parents:
        normalized = normalize_class(parent.name)
        if normalized:
            return normalized
    return None


def merge_all_datasets():
    """Main merge function"""
    print("="*60)
    print("SMART DATASET MERGER v3 (MinIO Integration)")
    print("="*60)
    
    # Step 1: Download from MinIO
    download_from_minio()
    
    # Step 2: Process staging directory
    print("\n" + "="*60)
    print("PROCESSING DATASETS")
    print("="*60)
    
    all_images = []
    
    # Process each dataset in staging
    for ds_dir in STAGING_DIR.iterdir():
        if not ds_dir.is_dir():
            continue
        
        print(f"\n[Processing] {ds_dir.name}")
        
        # Detect format
        has_yolo = ((ds_dir / "train" / "images").exists() and 
                   (ds_dir / "train" / "labels").exists())
        has_classification = False
        
        # Check for classification format
        for split in ["train", "valid", "test"]:
            split_dir = ds_dir / split
            if split_dir.exists():
                for item in split_dir.iterdir():
                    if item.is_dir() and normalize_class(item.name):
                        has_classification = True
                        break
        
        if has_yolo:
            print(f"  Format: YOLO")
            images = process_yolo_dataset(ds_dir)
        elif has_classification:
            print(f"  Format: Classification")
            images = process_classification_dataset(ds_dir)
        else:
            print(f"  Format: Unknown (treating as classification)")
            images = process_classification_dataset(ds_dir)
        
        print(f"  Found {len(images)} labeled images")
        
        # Count by class
        class_counts = defaultdict(int)
        for _, _, cls, _ in images:
            class_counts[cls] += 1
        
        for cls, count in sorted(class_counts.items()):
            print(f"    {cls}: {count}")
        
        all_images.extend(images)
    
    print(f"\n[Total] {len(all_images)} images across all datasets")
    
    if not all_images:
        print("[ERROR] No images found")
        return False
    
    # Split into train/val/test
    random.seed(42)
    random.shuffle(all_images)
    n = len(all_images)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    splits = {
        "train": all_images[:train_end],
        "val": all_images[train_end:val_end],
        "test": all_images[val_end:]
    }
    
    # Clear and create output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    
    # Write files
    print("\n[Writing] Creating YOLOv8 dataset...")
    
    for split_name, split_data in splits.items():
        img_dir = OUTPUT_DIR / split_name / "images"
        lbl_dir = OUTPUT_DIR / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (img_path, lbl_path, class_name, _) in enumerate(split_data):
            # Create unique name
            new_stem = f"{class_name}_{img_path.stem}_{i}"
            new_img = img_dir / f"{new_stem}{img_path.suffix}"
            new_lbl = lbl_dir / f"{new_stem}.txt"
            
            # Copy image
            shutil.copy2(img_path, new_img)
            
            # Create label
            class_id = CLASS_MAP[class_name]
            
            if lbl_path and lbl_path.exists():
                # Copy existing YOLO label
                shutil.copy2(lbl_path, new_lbl)
            else:
                # Create pseudo-label (full image bbox)
                new_lbl.write_text(f"{class_id} 0.5 0.5 1.0 1.0\n")
    
    # Create data.yaml
    yaml_path = OUTPUT_DIR / "data.yaml"
    data = {
        "path": str(OUTPUT_DIR.absolute()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: name for i, name in enumerate(CLASS_MAP.keys())},
        "nc": len(CLASS_MAP)
    }
    
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)
    
    # Upload final dataset to MinIO
    upload_to_minio()
    
    # Summary
    print("\n[SUCCESS] Dataset created!")
    print(f"  Location: {OUTPUT_DIR}")
    for split in ["train", "val", "test"]:
        count = len(list((OUTPUT_DIR / split / "images").glob("*")))
        print(f"  {split}: {count} images")
    
    # Final class distribution
    print("\n[Final class distribution]:")
    class_counts = defaultdict(int)
    for split in ["train", "val", "test"]:
        lbl_dir = OUTPUT_DIR / split / "labels"
        for lbl in lbl_dir.glob("*.txt"):
            try:
                with open(lbl) as f:
                    line = f.readline().strip()
                    if line:
                        cls_id = int(line.split()[0])
                        class_counts[list(CLASS_MAP.keys())[cls_id]] += 1
            except:
                pass
    
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")
    
    return True


def upload_to_minio():
    """Upload final dataset back to MinIO"""
    print("\n[Upload] Saving final dataset to MinIO...")
    
    client = Minio(
        MINIO_URL,
        access_key=MINIO_USER,
        secret_key=MINIO_PASSWORD,
        secure=False
    )
    
    bucket = "processed-datasets"
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
    
    uploaded = 0
    for root, _, files in os.walk(OUTPUT_DIR):
        for f in files:
            local_path = Path(root) / f
            relative_path = local_path.relative_to(OUTPUT_DIR)
            
            client.fput_object(
                bucket,
                f"yolo_dataset/{relative_path}",
                str(local_path)
            )
            uploaded += 1
    
    print(f"[OK] Uploaded {uploaded} files to MinIO bucket '{bucket}'")


if __name__ == "__main__":
    merge_all_datasets()
    