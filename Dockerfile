FROM apache/spark-py:latest
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --no-cache-dir uv

# Use stable Airflow 2.10.4
ENV AIRFLOW_VERSION=2.10.4

# Install Airflow with constraints
RUN uv pip install --system \
    "apache-airflow[postgres,amazon]==${AIRFLOW_VERSION}" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.10.txt"

# Install additional dependencies (including minio)
RUN uv pip install --system \
    opencv-python-headless \
    numpy \
    pandas \
    pyarrow \
    roboflow \
    pillow \
    tqdm \
    matplotlib \
    seaborn \
    scipy \
    scikit-image \
    psycopg2-binary \
    mlflow \
    ultralytics \
    torch \
    torchvision \
    pyyaml \
    pika \
    werkzeug \
    minio \
    apache-airflow-providers-apache-spark

# Set environment variables
ENV AIRFLOW_HOME=/workspace/airflow
ENV AIRFLOW__CORE__DAGS_FOLDER=/workspace/dags

# Create directories
RUN mkdir -p $AIRFLOW_HOME/dags && \
    mkdir -p /workspace/dags && \
    mkdir -p /workspace/dataset && \
    mkdir -p /workspace/source && \
    mkdir -p /workspace/config && \
    mkdir -p /workspace/experiments && \
    chmod -R 777 /workspace

WORKDIR /workspace
