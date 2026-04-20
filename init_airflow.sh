#!/bin/bash
set -e

echo "Initializing Airflow DB..."
airflow db init

echo "Creating admin user..."
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

echo "Creating MinIO connection..."
airflow connections add minio_default \
    --conn-type aws \
    --conn-login minioadmin \
    --conn-password minioadmin \
    --conn-extra '{"endpoint_url": "http://minio:9000"}' || true

echo "Airflow initialization complete!"
