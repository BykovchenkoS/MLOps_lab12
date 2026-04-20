"""Project Configuration"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_DIR = PROJECT_ROOT / 'source'
DATASET_DIR = PROJECT_ROOT / 'dataset'
DAGS_DIR = PROJECT_ROOT / 'dags'
EXPERIMENTS_DIR = PROJECT_ROOT / 'experiments'
MODELS_DIR = EXPERIMENTS_DIR / 'models'
METRICS_DIR = EXPERIMENTS_DIR / 'metrics'

# Source subdirectories
DOWNLOADING_DIR = SOURCE_DIR / 'downloading'
PREPROCESSING_DIR = SOURCE_DIR / 'preprocessing'
CLASSIFICATION_DIR = SOURCE_DIR / 'classification'

# Create experiment directories
EXPERIMENTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
METRICS_DIR.mkdir(exist_ok=True)

# API Keys
ROBOFLOW_API_KEY = 'rvWw2jzTZQW7A12hHzt2'

# MLflow Configuration
MLFLOW_TRACKING_URI = 'sqlite:///experiments/mlflow.db'
MLFLOW_EXPERIMENT_NAME = 'wildlife_classification'

# Dataset Configuration
DATASETS = [
    {
        'workspace': 'wildlife-classification',
        'project': 'wildlife-pqe9x',
        'version': 1,
        'format': 'folder',
        'name': 'wildlife-pqe9x'
    },
    {
        'workspace': 'trail-camera-wildlife',
        'project': 'trail-camera-wildlife-b0kkq',
        'version': 1,
        'format': 'folder',
        'name': 'trail-camera-wildlife-b0kkq'
    }
]

# Airflow Configuration
AIRFLOW_DAG_DEFAULTS = {
    'owner': 'mlops',
    'retries': 1,
    'retry_delay_minutes': 5,
}

# Image Extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', 
                   '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.WEBP'}

# Spark Configuration
SPARK_MASTER = 'local[*]'
SPARK_DRIVER_MEMORY = '2G'
SPARK_PACKAGES = [
    'org.apache.hadoop:hadoop-aws:3.3.4',
    'com.amazonaws:aws-java-sdk-bundle:1.12.262'
]

# YOLO Configuration - Optimized for fast fine-tuning
YOLO_MODEL = 'yolov8n-cls'  # nano classification model (much faster than medium)
YOLO_EPOCHS = 5  # Reduced from 10 for faster training
YOLO_IMG_SIZE = 224  # Reduced from 640 (typical for classification)
YOLO_BATCH_SIZE = 32  # Increased for better efficiency
YOLO_CONF_THRESHOLD = 0.5
YOLO_DEVICE = 'cpu'  # Use 'cpu' or '0' for GPU
YOLO_PATIENCE = 3  # Early stopping if no improvement for 3 epochs
