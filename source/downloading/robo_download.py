#!/usr/bin/env python3

import os
import shutil
import logging
from typing import List, Dict
from roboflow import Roboflow
from minio import Minio
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# SHARED PIPELINE PATH
# =========================

STAGING_DIR = Path("/workspace/datasets_staging")


class RoboflowDownloader:

    def __init__(self, api_key: str):

        self.rf = Roboflow(api_key=api_key)

        self.minio = Minio(
            "minio:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
            secure=False
        )

        if not self.minio.bucket_exists("datasets"):
            self.minio.make_bucket("datasets")

        # IMPORTANT: ensure shared folder exists
        STAGING_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    def get_export_format(self, project) -> str:
        if project.type == "object-detection":
            return "yolov8"
        if project.type == "classification":
            return "folder"
        return "folder"

    # ----------------------------
    def download_dataset(self, workspace: str, project_id: str, version: int):

        project = self.rf.workspace(workspace).project(project_id)

        export_format = self.get_export_format(project)

        logger.info(f"{project_id} → format={export_format}")

        dataset = project.version(version).download(export_format)

        return dataset.location, export_format

    # ----------------------------
    # FIXED: copy into shared staging dir
    # ----------------------------
    def move_to_staging(self, src_path: str, prefix: str):

        src = Path(src_path)

        if not src.exists():
            raise RuntimeError(f"Downloaded dataset not found: {src}")

        dst = STAGING_DIR / prefix

        if dst.exists():
            shutil.rmtree(dst)

        shutil.copytree(src, dst)

        logger.info(f"[OK] staged to {dst}")

        return dst

    # ----------------------------
    def upload(self, path: str, prefix: str):

        count = 0

        for root, _, files in os.walk(path):
            for f in files:

                full = os.path.join(root, f)
                rel = os.path.relpath(full, path)

                obj = f"{prefix}/{rel}"

                self.minio.fput_object("datasets", obj, full)

                count += 1

        logger.info(f"{prefix}: uploaded {count} files")

    # ----------------------------
    def run(self, datasets: List[Dict]):

        for ds in datasets:

            path, fmt = self.download_dataset(
                ds["workspace"],
                ds["project"],
                ds["version"]
            )

            # 🔥 CRITICAL FIX HERE
            staged = self.move_to_staging(path, ds["prefix"])

            self.upload(str(staged), f"{ds['prefix']}-{fmt}")


# =========================
DATASETS = [
    {
        "workspace": "majorproject-gmmfw",
        "project": "wildlife-a2ocs",
        "version": 1,
        "prefix": "wildlife"
    },
    {
        "workspace": "trail-camera-wildlife",
        "project": "trail-camera-wildlife-b0kkq",
        "version": 4,
        "prefix": "trail-camera"
    }
]


def main():

    downloader = RoboflowDownloader(
        api_key=os.getenv("ROBOFLOW_API_KEY", "rvWw2jzTZQW7A12hHzt2")
    )

    downloader.run(DATASETS)


if __name__ == "__main__":
    main()
