"""
Complete MLOps Pipeline DAG (STABLE YOLO VERSION)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from pathlib import Path
import json

default_args = {
    'owner': 'mlops',
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
    'start_date': datetime(2024, 1, 1),
}

dag = DAG(
    'mlops_complete_pipeline',
    default_args=default_args,
    description='Stable YOLO pipeline',
    schedule='@monthly',
    catchup=False,
    tags=['mlops', 'yolo', 'stable'],
    max_active_runs=1,   
    concurrency=1, 
)


# =========================
# 1. DOWNLOAD + MERGE
# =========================

download_task = BashOperator(
    task_id='download_datasets',
    bash_command='cd /workspace && uv run python source/downloading/robo_download.py',
    dag=dag,
)

merge_task = BashOperator(
    task_id='merge_datasets',
    bash_command='cd /workspace && uv run python source/preprocessing/merge_and_upload_datasets.py',
    dag=dag,
)


# =========================
# 2. YOLO DATASET FINALIZATION (REQUIRED FIX)
# =========================

finalize_dataset = BashOperator(
    task_id='finalize_dataset',
    bash_command=(
        'cd /workspace && '
        'uv run python source/preprocessing/yolo_finalize_dataset.py'
    ),
    dag=dag,
)


# =========================
# 3. PRE-CLEAN BASELINE
# =========================

baseline_pre = BashOperator(
    task_id='baseline_pre_clean',
    bash_command=(
        'cd /workspace && '
        'uv run python source/classification/yolo_models.py baseline '
        '--dataset /workspace/dataset_yolo --stage pre_clean'
    ),
    dag=dag,
)

heatmap_pre = BashOperator(
    task_id='heatmap_pre_clean',
    bash_command=(
        'cd /workspace && '
        'uv run python source/classification/yolo_models.py heatmap '
        '--stage pre_clean'
    ),
    dag=dag,
)


# =========================
# 4. CLEAN DATASET (OPTIONAL BUT SAFE)
# =========================

clean_task = BashOperator(
    task_id='clean_dataset',
    bash_command='cd /workspace && uv run python source/preprocessing/clean_dataset.py /workspace/dataset_yolo --execute',
    dag=dag,
)


# =========================
# 5. POST-CLEAN BASELINE
# =========================

baseline_post = BashOperator(
    task_id='baseline_post_clean',
    bash_command=(
        'cd /workspace && '
        'uv run python source/classification/yolo_models.py baseline '
        '--dataset /workspace/dataset_yolo --stage post_clean'
    ),
    dag=dag,
)

heatmap_post = BashOperator(
    task_id='heatmap_post_clean',
    bash_command=(
        'cd /workspace && '
        'uv run python source/classification/yolo_models.py heatmap '
        '--stage post_clean'
    ),
    dag=dag,
)

compare_baselines = BashOperator(
    task_id='compare_baselines',
    bash_command='cd /workspace && uv run python source/classification/yolo_models.py compare-baselines',
    dag=dag,
)


# =========================
# 6. TRAIN REFINED MODEL
# =========================

train_refined = BashOperator(
    task_id='train_refined',
    bash_command=(
        'cd /workspace && '
        'uv run python source/classification/yolo_models.py refined '
        '--dataset /workspace/dataset_yolo --epochs 25 --stage refined'
    ),
    dag=dag,
)

heatmap_refined = BashOperator(
    task_id='heatmap_refined',
    bash_command=(
        'cd /workspace && '
        'uv run python source/classification/yolo_models.py heatmap '
        '--stage refined'
    ),
    dag=dag,
)


# =========================
# 7. FINAL ANALYSIS
# =========================

final_compare = BashOperator(
    task_id='compare_all',
    bash_command='cd /workspace && uv run python source/classification/yolo_models.py compare-all',
    dag=dag,
)

report_task = BashOperator(
    task_id='generate_report',
    bash_command='cd /workspace && uv run python source/classification/yolo_models.py report',
    dag=dag,
)


# =========================
# DEPENDENCIES (FIXED FLOW)
# =========================

download_task >> merge_task >> finalize_dataset

finalize_dataset >> baseline_pre >> heatmap_pre

heatmap_pre >> clean_task

clean_task >> baseline_post >> heatmap_post

heatmap_post >> compare_baselines

compare_baselines >> train_refined >> heatmap_refined

heatmap_refined >> final_compare >> report_task
