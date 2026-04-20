#!/usr/bin/env python3
"""
YOLO Models - With MLflow Tracking, Training Plots, and MinIO Integration
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import yaml
import mlflow
import pandas as pd
import os
from minio import Minio

# =========================
# MINIO CONFIGURATION
# =========================

MINIO_URL = os.environ.get("MINIO_URL", "minio:9000")
MINIO_USER = os.environ.get("MINIO_USER", "minioadmin")
MINIO_PASSWORD = os.environ.get("MINIO_PASSWORD", "minioadmin")

# MLflow S3/MinIO configuration for artifact storage
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{MINIO_URL}"
os.environ["AWS_ACCESS_KEY_ID"] = MINIO_USER
os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_PASSWORD

# =========================
# CONFIGURATION
# =========================

YOLO_DEVICE = "mps" if sys.platform == "darwin" else "cpu"
DEFAULT_EPOCHS = 20

# MLflow configuration
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# =========================
# METRICS IO
# =========================

METRICS_DIR = Path("experiments/metrics")
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = Path("experiments/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def get_minio_client():
    """Create and return MinIO client"""
    return Minio(
        MINIO_URL,
        access_key=MINIO_USER,
        secret_key=MINIO_PASSWORD,
        secure=False
    )


def download_dataset_from_minio(dataset_name: str = "yolo_dataset"):
    """Pull dataset from MinIO if not present locally"""
    dataset_path = Path("/workspace/dataset_yolo")
    
    # If dataset already exists with images, skip
    if dataset_path.exists():
        images = list(dataset_path.rglob("*.jpg")) + list(dataset_path.rglob("*.png"))
        if images:
            print(f"[OK] Dataset already exists at {dataset_path} ({len(images)} images)")
            return dataset_path
    
    print(f"[*] Downloading dataset from MinIO...")
    
    client = get_minio_client()
    bucket = "processed-datasets"
    
    if not client.bucket_exists(bucket):
        print(f"[WARN] Bucket '{bucket}' does not exist in MinIO")
        return dataset_path
    
    prefix = f"{dataset_name}/"
    objects = list(client.list_objects(bucket, prefix=prefix, recursive=True))
    
    if not objects:
        print(f"[WARN] No objects found in {bucket}/{prefix}")
        return dataset_path
    
    downloaded = 0
    for obj in objects:
        rel_path = obj.object_name.replace(prefix, "")
        if not rel_path:  # Skip directories
            continue
        local_path = dataset_path / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        client.fget_object(bucket, obj.object_name, str(local_path))
        downloaded += 1
    
    print(f"[OK] Downloaded {downloaded} files from MinIO to {dataset_path}")
    return dataset_path


def upload_model_to_minio(model_path: Path, stage: str) -> str:
    """Upload trained model to MinIO models bucket"""
    client = get_minio_client()
    
    bucket = "models"
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        print(f"[OK] Created bucket: {bucket}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    object_name = f"yolo_{stage}/{timestamp}/best.pt"
    
    client.fput_object(bucket, object_name, str(model_path))
    minio_uri = f"s3://{bucket}/{object_name}"
    print(f"[OK] Model uploaded to MinIO: {minio_uri}")
    
    return minio_uri


def upload_experiment_artifacts(stage: str):
    """Upload experiment results to MinIO"""
    client = get_minio_client()
    
    bucket = "experiments"
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"yolo_{stage}/{timestamp}"
    
    uploaded = 0
    for local_dir, remote_prefix in [
        (METRICS_DIR, "metrics"),
        (PLOTS_DIR, "plots"),
        (Path("experiments/heatmaps"), "heatmaps"),
    ]:
        if local_dir.exists():
            for file_path in local_dir.glob("*"):
                if file_path.is_file():
                    object_name = f"{prefix}/{remote_prefix}/{file_path.name}"
                    client.fput_object(bucket, object_name, str(file_path))
                    uploaded += 1
    
    if uploaded > 0:
        print(f"[OK] Uploaded {uploaded} artifacts to MinIO bucket '{bucket}'")


def get_dataset_path():
    """Get dataset path, downloading from MinIO if needed"""
    return download_dataset_from_minio()


def save_metrics(metrics: dict, stage: str, model_type: str):
    """Save metrics with consistent naming"""
    path = METRICS_DIR / f"{model_type}_{stage}_metrics.json"
    
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        if hasattr(obj, 'item'):
            return obj.item()
        return str(obj)
    
    clean_metrics = json.loads(json.dumps(metrics, default=convert))
    
    with open(path, "w") as f:
        json.dump(clean_metrics, f, indent=2)
    print(f"[OK] Metrics saved: {path}")
    return path


def load_metrics(model_type: str, stage: str):
    """Load metrics with explicit model_type and stage"""
    path = METRICS_DIR / f"{model_type}_{stage}_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_latest_metrics(stage: str):
    """Fallback loader that tries both baseline and refined"""
    for model_type in ["baseline", "refined"]:
        metrics = load_metrics(model_type, stage)
        if metrics:
            return metrics
    return None


# =========================
# DATASET VALIDATION
# =========================

def validate_dataset(dataset_path: Path):
    """Ensure dataset has required structure before training"""
    dataset_path = Path(dataset_path)
    
    required_dirs = [
        dataset_path / "train" / "images",
        dataset_path / "val" / "images",
    ]
    
    for d in required_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Missing required directory: {d}")
    
    train_images = list((dataset_path / "train" / "images").glob("*"))
    val_images = list((dataset_path / "val" / "images").glob("*"))
    
    # Filter for image files only
    train_images = [f for f in train_images if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    val_images = [f for f in val_images if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    if not train_images:
        raise ValueError(f"No training images found in {dataset_path}/train/images")
    if not val_images:
        raise ValueError(f"No validation images found in {dataset_path}/val/images")
    
    print(f"[OK] Dataset validated: {len(train_images)} train, {len(val_images)} val images")
    
    return {"train_count": len(train_images), "val_count": len(val_images)}


# =========================
# YAML GENERATION
# =========================

def ensure_yolo_yaml(dataset_path: Path):
    """Create YOLO data.yaml with correct class mapping"""
    
    dataset_path = Path(dataset_path)
    yaml_path = dataset_path / "data.yaml"
    
    CLASS_NAMES = ["Bear", "Boar", "Deer", "Elephant", "Tapir", "Tiger"]
    
    if yaml_path.exists():
        with open(yaml_path) as f:
            existing = yaml.safe_load(f)
            if existing.get("names") and len(existing["names"]) == len(CLASS_NAMES):
                return yaml_path
    
    data = {
        "path": str(dataset_path.absolute()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images" if (dataset_path / "test").exists() else "val/images",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
        "nc": len(CLASS_NAMES)
    }
    
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)
    
    print(f"[OK] Created YOLO data.yaml with {len(CLASS_NAMES)} classes")
    return yaml_path


# =========================
# HELPER: Safe value extraction
# =========================

def safe_extract_scalar(val):
    """Safely extract a scalar from tensor, numpy array, or list"""
    if val is None:
        return 0.0
    
    try:
        if isinstance(val, np.ndarray):
            if val.size == 0:
                return 0.0
            return float(val.mean()) if val.size > 1 else float(val.item())
        
        if hasattr(val, 'numel'):
            if val.numel() == 0:
                return 0.0
            return float(val.mean().item()) if val.numel() > 1 else float(val.item())
        
        if isinstance(val, (list, tuple)):
            if len(val) == 0:
                return 0.0
            return float(sum(val) / len(val))
        
        return float(val)
    except:
        return 0.0


# =========================
# TRAINING PLOTS
# =========================

def generate_training_plots(results_dir: Path, stage: str):
    """Generate training loss and metric plots from results.csv"""
    
    results_csv = results_dir / "results.csv"
    if not results_csv.exists():
        print(f"[WARN] No results.csv found in {results_dir}")
        return None
    
    df = pd.read_csv(results_csv)
    
    # Clean column names (remove whitespace)
    df.columns = df.columns.str.strip()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Loss curves
    ax1 = axes[0, 0]
    if 'train/box_loss' in df.columns:
        ax1.plot(df.index, df['train/box_loss'], label='Train Box Loss', linewidth=2)
    if 'train/cls_loss' in df.columns:
        ax1.plot(df.index, df['train/cls_loss'], label='Train Class Loss', linewidth=2)
    if 'val/box_loss' in df.columns:
        ax1.plot(df.index, df['val/box_loss'], label='Val Box Loss', linewidth=2)
    if 'val/cls_loss' in df.columns:
        ax1.plot(df.index, df['val/cls_loss'], label='Val Class Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss Curves - {stage}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: mAP curves
    ax2 = axes[0, 1]
    if 'metrics/mAP50(B)' in df.columns:
        ax2.plot(df.index, df['metrics/mAP50(B)'], label='mAP50', linewidth=2, marker='o')
    if 'metrics/mAP50-95(B)' in df.columns:
        ax2.plot(df.index, df['metrics/mAP50-95(B)'], label='mAP50-95', linewidth=2, marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mAP')
    ax2.set_title(f'mAP Curves - {stage}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Precision & Recall
    ax3 = axes[1, 0]
    if 'metrics/precision(B)' in df.columns:
        ax3.plot(df.index, df['metrics/precision(B)'], label='Precision', linewidth=2, marker='o')
    if 'metrics/recall(B)' in df.columns:
        ax3.plot(df.index, df['metrics/recall(B)'], label='Recall', linewidth=2, marker='s')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.set_title(f'Precision & Recall - {stage}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Learning Rate
    ax4 = axes[1, 1]
    if 'lr/pg0' in df.columns:
        ax4.plot(df.index, df['lr/pg0'], label='Learning Rate', linewidth=2, color='green')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title(f'Learning Rate Schedule - {stage}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    
    plot_path = PLOTS_DIR / f"training_curves_{stage}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Training plots saved: {plot_path}")
    return plot_path


# =========================
# EVALUATION
# =========================

def evaluate_model_per_class(model, dataset_path, model_name):
    print(f"\n[*] Evaluating {model_name}")
    
    dataset_path = Path(dataset_path)
    yaml_path = ensure_yolo_yaml(dataset_path)
    
    try:
        results = model.val(
            data=str(yaml_path),
            device=YOLO_DEVICE,
            verbose=False
        )
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        return {
            "name": model_name,
            "precision": 0.0,
            "recall": 0.0,
            "map50": 0.0,
            "map50_95": 0.0,
            "per_class_ap": {},
            "class_names": {},
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }
    
    with open(yaml_path) as f:
        class_names = yaml.safe_load(f).get("names", {})
    
    per_class = {}
    
    if hasattr(results, "ap_class_index") and hasattr(results, "ap"):
        ap_array = results.ap
        for idx, ap in zip(results.ap_class_index, ap_array):
            ap_value = safe_extract_scalar(ap)
            class_name = class_names.get(int(idx), f"class_{idx}")
            per_class[class_name] = ap_value
    
    box = results.box if hasattr(results, 'box') else None
    
    precision = 0.0
    recall = 0.0
    map50 = 0.0
    map50_95 = 0.0
    
    if box is not None:
        if hasattr(box, 'p') and box.p is not None:
            precision = safe_extract_scalar(box.p)
        if hasattr(box, 'r') and box.r is not None:
            recall = safe_extract_scalar(box.r)
        if hasattr(box, 'map50') and box.map50 is not None:
            map50 = safe_extract_scalar(box.map50)
        if hasattr(box, 'map') and box.map is not None:
            map50_95 = safe_extract_scalar(box.map)
    
    if hasattr(results, 'results_dict'):
        rd = results.results_dict
        if precision == 0.0:
            precision = float(rd.get('metrics/precision(B)', 0.0))
        if recall == 0.0:
            recall = float(rd.get('metrics/recall(B)', 0.0))
        if map50 == 0.0:
            map50 = float(rd.get('metrics/mAP50(B)', 0.0))
        if map50_95 == 0.0:
            map50_95 = float(rd.get('metrics/mAP50-95(B)', 0.0))
    
    return {
        "name": model_name,
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
        "per_class_ap": per_class,
        "class_names": class_names,
        "timestamp": datetime.now().isoformat()
    }


# =========================
# HEATMAPS
# =========================

def generate_heatmap(metrics, stage):
    out_dir = Path("experiments/heatmaps")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    per_class = metrics.get("per_class_ap", {})
    if not per_class:
        print(f"[WARN] No class metrics for {stage}")
        return None
    
    sorted_cls = sorted(per_class.items(), key=lambda x: x[1], reverse=True)
    names = [c[0][:20] for c in sorted_cls]
    vals = [c[1] for c in sorted_cls]
    
    fig, ax = plt.subplots(figsize=(12, max(5, len(names) * 0.3)))
    
    colors = plt.cm.RdYlGn(np.array(vals))
    bars = ax.barh(names, vals, color=colors)
    ax.set_title(f"Per-Class Average Precision - {stage}")
    ax.set_xlabel("AP")
    ax.set_xlim([0, 1.0])
    
    for i, (bar, v) in enumerate(zip(bars, vals)):
        ax.text(v + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{v:.3f}", va="center", fontsize=8)
    
    path = out_dir / f"heatmap_{stage}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Heatmap saved: {path}")
    return path


# =========================
# BASELINE
# =========================

def run_baseline(dataset_path, stage):
    dataset_path = Path(dataset_path)
    
    print(f"\n===== BASELINE [{stage}] =====")
    
    # Start MLflow run
    mlflow.set_experiment("wildlife-yolo-baseline")
    
    with mlflow.start_run(run_name=f"baseline_{stage}"):
        # Log parameters
        mlflow.log_params({
            "stage": stage,
            "model": "yolov8n",
            "device": YOLO_DEVICE,
            "dataset": str(dataset_path)
        })
        
        # Validate dataset
        counts = validate_dataset(dataset_path)
        mlflow.log_metrics({
            "train_images": counts["train_count"],
            "val_images": counts["val_count"]
        })
        
        # Load pretrained model
        model = YOLO("yolov8n.pt")
        
        metrics = evaluate_model_per_class(model, dataset_path, f"baseline_{stage}")
        metrics["stage"] = stage
        metrics["model_type"] = "baseline"
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "map50": metrics["map50"],
            "map50_95": metrics["map50_95"]
        })
        
        # Log per-class metrics
        for cls, ap in metrics["per_class_ap"].items():
            mlflow.log_metric(f"ap_{cls}", ap)
        
        save_metrics(metrics, stage, "baseline")
        heatmap_path = generate_heatmap(metrics, stage)
        
        if heatmap_path:
            mlflow.log_artifact(str(heatmap_path))
        
        # Upload artifacts to MinIO
        upload_experiment_artifacts(f"{stage}_baseline")
        
        print(f"\n[SUMMARY] Baseline Results:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  mAP50:     {metrics['map50']:.4f}")
        print(f"  mAP50-95:  {metrics['map50_95']:.4f}")
        print(f"\n[MLflow] Run ID: {mlflow.active_run().info.run_id}")
    
    return metrics


# =========================
# REFINED/TRAINING
# =========================

def run_refined(dataset_path, epochs, stage):
    dataset_path = Path(dataset_path)
    
    print(f"\n===== REFINE/TRAIN [{stage}] =====")
    
    # Start MLflow run
    mlflow.set_experiment("wildlife-yolo-refined")
    
    with mlflow.start_run(run_name=f"refined_{stage}_{epochs}ep"):
        # Log parameters
        mlflow.log_params({
            "stage": stage,
            "model": "yolov8n",
            "epochs": epochs,
            "device": YOLO_DEVICE,
            "dataset": str(dataset_path),
            "imgsz": 640
        })
        
        # Validate dataset
        counts = validate_dataset(dataset_path)
        mlflow.log_metrics({
            "train_images": counts["train_count"],
            "val_images": counts["val_count"]
        })
        
        # Ensure YAML exists
        yaml_path = ensure_yolo_yaml(dataset_path)
        
        # Load pretrained model
        model = YOLO("yolov8n.pt")
        
        # Train
        print(f"[*] Training for {epochs} epochs...")
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=640,
            device=YOLO_DEVICE,
            project="experiments",
            name=f"yolo_{stage}",
            exist_ok=True,
            verbose=True,
            patience=5  # Early stopping
        )
        
        # Generate training plots
        results_dir = Path(f"experiments/yolo_{stage}")
        plot_path = generate_training_plots(results_dir, stage)
        if plot_path:
            mlflow.log_artifact(str(plot_path))
        
        # Load best weights
        best_path = Path(f"experiments/yolo_{stage}/weights/best.pt")
        if not best_path.exists():
            raise FileNotFoundError(f"Training failed: {best_path} not found")
        
        # Upload model to MinIO
        minio_model_path = upload_model_to_minio(best_path, stage)
        mlflow.log_param("minio_model_path", minio_model_path)
        
        # Log model artifact to MLflow
        mlflow.log_artifact(str(best_path))
        
        best_model = YOLO(str(best_path))
        
        # Evaluate
        metrics = evaluate_model_per_class(best_model, dataset_path, f"refined_{stage}")
        metrics["stage"] = stage
        metrics["model_type"] = "refined"
        metrics["epochs"] = epochs
        metrics["minio_model_path"] = minio_model_path
        
        # Log final metrics
        mlflow.log_metrics({
            "final_precision": metrics["precision"],
            "final_recall": metrics["recall"],
            "final_map50": metrics["map50"],
            "final_map50_95": metrics["map50_95"]
        })
        
        # Log per-class metrics
        for cls, ap in metrics["per_class_ap"].items():
            mlflow.log_metric(f"final_ap_{cls}", ap)
        
        save_metrics(metrics, stage, "refined")
        heatmap_path = generate_heatmap(metrics, stage)
        
        if heatmap_path:
            mlflow.log_artifact(str(heatmap_path))
        
        # Upload artifacts to MinIO
        upload_experiment_artifacts(f"{stage}_refined")
        
        print(f"\n[SUMMARY] Refined Results:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  mAP50:     {metrics['map50']:.4f}")
        print(f"  mAP50-95:  {metrics['map50_95']:.4f}")
        print(f"  Model:     {minio_model_path}")
        print(f"\n[MLflow] Run ID: {mlflow.active_run().info.run_id}")
    
    return metrics


# =========================
# COMPARISONS
# =========================

def compare_baselines():
    """Compare pre vs post clean baselines"""
    pre = load_metrics("baseline", "pre_clean")
    post = load_metrics("baseline", "post_clean")
    
    if not pre or not post:
        print("[ERROR] Missing baseline metrics. Run baseline evaluations first.")
        return
    
    labels = ["Precision", "Recall", "mAP50", "mAP50-95"]
    
    pre_vals = [pre.get(k, 0) for k in ["precision", "recall", "map50", "map50_95"]]
    post_vals = [post.get(k, 0) for k in ["precision", "recall", "map50", "map50_95"]]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, pre_vals, width, label="Pre-clean", color='skyblue')
    ax.bar(x + width/2, post_vals, width, label="Post-clean", color='lightcoral')
    
    ax.set_ylabel('Score')
    ax.set_title('Baseline Model Comparison: Pre vs Post Clean')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim([0, 1.0])
    
    out = Path("experiments/comparisons")
    out.mkdir(parents=True, exist_ok=True)
    
    path = out / "baseline_compare.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    
    print(f"[OK] Saved comparison: {path}")
    
    # Log to MLflow
    mlflow.set_experiment("wildlife-yolo-comparisons")
    with mlflow.start_run(run_name="baseline_comparison"):
        mlflow.log_metrics({
            "pre_clean_map50": pre.get("map50", 0),
            "post_clean_map50": post.get("map50", 0),
            "improvement": post.get("map50", 0) - pre.get("map50", 0)
        })
        mlflow.log_artifact(str(path))


# =========================
# CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Pipeline with MLflow and MinIO")
    parser.add_argument("command",
        choices=["baseline", "refined", "heatmap", "compare-baselines", "validate", "download"])
    parser.add_argument("--dataset", default=None, help="Dataset path (auto-download from MinIO if not provided)")
    parser.add_argument("--stage", default="pre_clean", help="Stage name")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs")
    
    args = parser.parse_args()
    
    # Get dataset path (download from MinIO if needed)
    if args.dataset:
        dataset_path = Path(args.dataset)
    else:
        dataset_path = get_dataset_path()
    
    print(f"\n[INFO] Using dataset: {dataset_path}")
    print(f"[INFO] Device: {YOLO_DEVICE}")
    print(f"[INFO] MLflow URI: {MLFLOW_TRACKING_URI}")
    
    if args.command == "download":
        # Just download dataset
        download_dataset_from_minio()
        
    elif args.command == "validate":
        validate_dataset(dataset_path)
        
    elif args.command == "baseline":
        run_baseline(dataset_path, args.stage)
        
    elif args.command == "refined":
        run_refined(dataset_path, args.epochs, args.stage)
        
    elif args.command == "heatmap":
        m = load_latest_metrics(args.stage)
        if m:
            generate_heatmap(m, args.stage)
        else:
            print(f"[ERROR] No metrics found for stage: {args.stage}")
            
    elif args.command == "compare-baselines":
        compare_baselines()
        