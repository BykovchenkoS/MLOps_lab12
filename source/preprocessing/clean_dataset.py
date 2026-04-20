#!/usr/bin/env python3
"""
Dataset cleaning utilities for removing duplicate augmentations
Integrated with DAG pipeline paths
"""

import re
import shutil
import sys
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Optional
from datetime import datetime

# Try to import tqdm, fallback if not available
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        print(f"[*] {desc}...")
        return iterable

# =========================
# CONFIGURATION
# =========================

# Match your DAG pipeline paths
DEFAULT_DATASET_PATH = Path("/workspace/dataset_yolo")
STAGING_PATH = Path("/workspace/datasets_staging")

# Image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

# Cleanup report path
REPORT_DIR = Path("/workspace/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# CORE FUNCTIONS
# =========================

def is_roboflow_augmented(filename) -> bool:
    """Check if file is a Roboflow augmented version"""
    filename_str = str(filename).lower()
    # Roboflow patterns: .rf.hex.jpg, _aug_*, etc.
    patterns = ['.rf.', '_aug_', '_flip', '_rotate', '_brightness', '_blur']
    return any(pattern in filename_str for pattern in patterns)


def get_base_image_name(filename) -> str:
    """Extract base image name without augmentation suffix"""
    filename = str(filename)
    stem = Path(filename).stem
    
    # Handle .rf.hex pattern
    match = re.match(r'^(.+?)\.rf\.[a-f0-9]+$', stem, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Handle other augmentation patterns
    aug_patterns = ['_aug', '_flip', '_rotate', '_brightness', '_blur']
    for pattern in aug_patterns:
        if pattern in stem:
            base = stem.split(pattern)[0]
            return base
    
    return stem


def find_class_for_image(img_path: Path, dataset_path: Path) -> str:
    """Find the class name for an image by traversing up to class folder"""
    current = img_path.parent
    
    # Folders to skip (not class names)
    skip_folders = {'train', 'test', 'val', 'valid', 'validation', 
                    'images', 'labels', 'annotations'}
    
    while current != dataset_path and current.parent != current:
        folder_name = current.name
        if folder_name.lower() not in skip_folders:
            return folder_name
        current = current.parent
    
    return "unknown"


def scan_dataset(dataset_path: Path) -> Dict[Tuple[str, str], List[Path]]:
    """Scan dataset and group images by (class, base_name)"""
    dataset_path = Path(dataset_path)
    class_base_to_versions = defaultdict(list)
    
    print(f"\n[*] Scanning: {dataset_path}")
    
    all_images = []
    for ext in IMAGE_EXTENSIONS:
        all_images.extend(dataset_path.rglob(f'*{ext}'))
        all_images.extend(dataset_path.rglob(f'*{ext.upper()}'))
    
    for img_file in tqdm(all_images, desc="Scanning images"):
        class_name = find_class_for_image(img_file, dataset_path)
        base_name = get_base_image_name(img_file.name)
        key = (class_name, base_name)
        class_base_to_versions[key].append(img_file)
    
    return class_base_to_versions


def get_dataset_stats(class_base_to_versions: Dict) -> Dict:
    """Calculate dataset statistics"""
    total_bases = len(class_base_to_versions)
    total_files = sum(len(versions) for versions in class_base_to_versions.values())
    
    duplicates = 0
    augmented_files = 0
    
    for versions in class_base_to_versions.values():
        if len(versions) > 1:
            duplicates += len(versions) - 1
        for f in versions:
            if is_roboflow_augmented(f):
                augmented_files += 1
    
    return {
        "unique_bases": total_bases,
        "total_files": total_files,
        "augmented_files": augmented_files,
        "files_to_remove": duplicates,
        "retention_rate": total_bases / total_files if total_files > 0 else 0
    }


def clean_roboflow_augmentations(dataset_path: Path, dry_run: bool = True) -> Dict:
    """Remove Roboflow augmented images keeping only originals"""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        return {"error": f"Dataset path not found: {dataset_path}", "removed": 0}
    
    # Scan dataset
    class_base_to_versions = scan_dataset(dataset_path)
    stats = get_dataset_stats(class_base_to_versions)
    
    print(f"\n[STATS] Dataset Statistics:")
    print(f"   Unique base images: {stats['unique_bases']}")
    print(f"   Total files: {stats['total_files']}")
    print(f"   Augmented files: {stats['augmented_files']}")
    print(f"   Files to remove: {stats['files_to_remove']}")
    print(f"   Retention rate: {stats['retention_rate']:.1%}")
    
    if stats['files_to_remove'] == 0:
        print("\n[OK] No duplicates or augmentations found")
        return {"action": "none", "removed": 0, **stats}
    
    if dry_run:
        print("\n[DRY RUN] No files will be deleted")
        print("   Use --execute to actually remove files")
        return {"action": "dry_run", "removed": 0, **stats}
    
    print(f"\n[CLEANING] Removing {stats['files_to_remove']} duplicate/augmented files...")
    removed_count = 0
    removed_by_class = defaultdict(int)
    
    for (class_name, base_name), versions in tqdm(
        class_base_to_versions.items(), 
        desc="Cleaning"
    ):
        if len(versions) <= 1:
            continue
        
        # Sort: keep original (non-augmented) first
        versions.sort(key=lambda x: (is_roboflow_augmented(x), str(x)))
        keep = versions[0]
        to_remove = versions[1:]
        
        for img_path in to_remove:
            try:
                # Remove image
                img_path.unlink()
                removed_count += 1
                removed_by_class[class_name] += 1
                
                # Remove corresponding label if exists
                label_path = img_path.with_suffix('.txt')
                if label_path.exists():
                    label_path.unlink()
                
                # Try to remove parent if empty
                parent = img_path.parent
                if parent != dataset_path and not any(parent.iterdir()):
                    parent.rmdir()
                    
            except Exception as e:
                print(f"   [WARN] Error deleting {img_path}: {e}")
    
    print(f"\n[OK] Removed {removed_count} files")
    
    # Count remaining
    remaining_stats = get_dataset_stats(scan_dataset(dataset_path))
    
    result = {
        "action": "executed",
        "removed": removed_count,
        "removed_by_class": dict(removed_by_class),
        "before": stats,
        "after": remaining_stats
    }
    
    return result


def count_originals_per_class(dataset_path: Path) -> Dict[str, int]:
    """Count unique original images per class"""
    class_base_to_versions = scan_dataset(dataset_path)
    
    class_counts = defaultdict(int)
    for (class_name, base_name), versions in class_base_to_versions.items():
        # Count each base image once per class
        class_counts[class_name] += 1
    
    print(f"\n[CLASS COUNTS] Original images per class:")
    print(f"{'Class':<30} {'Count':>10}")
    print("-" * 42)
    
    sorted_classes = sorted(class_counts.items(), key=lambda x: -x[1])
    for cls, count in sorted_classes:
        bar = '#' * min(count // 5, 50)
        print(f"{cls:<30} {count:>10} {bar}")
    
    total = sum(class_counts.values())
    print("-" * 42)
    print(f"{'TOTAL':<30} {total:>10}")
    
    return dict(class_counts)


def save_cleanup_report(result: Dict, dataset_path: Path) -> Path:
    """Save cleanup report to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"cleanup_report_{timestamp}.json"
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset_path": str(dataset_path),
        "result": result
    }
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"[OK] Report saved: {report_path}")
    return report_path


# =========================
# DAG INTEGRATION
# =========================

def run_cleanup_for_dag(dataset_path: Path = None, execute: bool = False) -> Dict:
    """
    Run cleanup and return results dict (for DAG integration)
    """
    if dataset_path is None:
        dataset_path = DEFAULT_DATASET_PATH
    
    dataset_path = Path(dataset_path)
    
    print("\n" + "="*60)
    print("DATASET CLEANUP")
    print("="*60)
    
    # Show current state
    print("\n[CURRENT STATE]")
    original_counts = count_originals_per_class(dataset_path)
    
    # Run cleanup
    print("\n[CLEANUP]")
    result = clean_roboflow_augmentations(dataset_path, dry_run=not execute)
    
    # Save report
    report_path = save_cleanup_report(result, dataset_path)
    result["report_path"] = str(report_path)
    
    # Show final state if executed
    if execute and result.get("removed", 0) > 0:
        print("\n[FINAL STATE]")
        final_counts = count_originals_per_class(dataset_path)
        result["final_class_counts"] = final_counts
    
    return result


# =========================
# CLI
# =========================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean Roboflow augmentations from dataset')
    parser.add_argument('dataset_path', nargs='?', default=str(DEFAULT_DATASET_PATH), 
                       help='Path to dataset directory')
    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='Preview changes without deleting (default)')
    parser.add_argument('--execute', action='store_true',
                       help='Actually delete files (use with caution)')
    parser.add_argument('--json', action='store_true',
                       help='Output results as JSON')
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"[ERROR] Dataset path not found: {dataset_path}")
        sys.exit(1)
    
    # Show current state
    print("\n" + "="*60)
    print("CURRENT DATASET STATE")
    print("="*60)
    original_counts = count_originals_per_class(dataset_path)
    
    # Preview cleanup
    print("\n" + "="*60)
    print("CLEANUP PREVIEW")
    print("="*60)
    result = clean_roboflow_augmentations(dataset_path, dry_run=True)
    
    # Ask for confirmation
    if args.execute:
        response = 'y'
    else:
        print("\n" + "="*60)
        response = input("Delete duplicates and augmentations? (y/n): ").strip().lower()
    
    if response == 'y':
        print("\n" + "="*60)
        print("EXECUTING CLEANUP")
        print("="*60)
        result = clean_roboflow_augmentations(dataset_path, dry_run=False)
        
        # Save report
        report_path = save_cleanup_report(result, dataset_path)
        
        # Show final state
        print("\n" + "="*60)
        print("FINAL DATASET STATE")
        print("="*60)
        final_counts = count_originals_per_class(dataset_path)
        print("\n[OK] Cleanup completed")
        
        if args.json:
            result["final_class_counts"] = final_counts
            result["report_path"] = str(report_path)
            print(json.dumps(result, indent=2, default=str))
    else:
        print("\n[*] Cleanup skipped. Use --execute to run without prompt.")
        if args.json:
            print(json.dumps(result, indent=2, default=str))
            