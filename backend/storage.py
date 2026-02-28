"""Filesystem-backed persistence for jobs, lane outputs, and deployments."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ARTIFACTS_ROOT = Path(os.getenv("DEPLOYBENCH_ARTIFACTS_ROOT", "/artifacts"))
JOBS_DIR = ARTIFACTS_ROOT / "jobs"
DEPLOYMENTS_DIR = ARTIFACTS_ROOT / "deployments"
MODELS_DIR = ARTIFACTS_ROOT / "models"
SAMPLES_DIR = ARTIFACTS_ROOT / "samples"
PREVIEWS_DIR = ARTIFACTS_ROOT / "previews"


@dataclass(frozen=True)
class JobPaths:
    job_id: str

    @property
    def job_dir(self) -> Path:
        return JOBS_DIR / self.job_id

    @property
    def job_json(self) -> Path:
        return self.job_dir / "job.json"

    @property
    def lanes_dir(self) -> Path:
        return self.job_dir / "lanes"



def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()



def ensure_dirs() -> None:
    for p in [ARTIFACTS_ROOT, JOBS_DIR, DEPLOYMENTS_DIR, MODELS_DIR, SAMPLES_DIR, PREVIEWS_DIR]:
        p.mkdir(parents=True, exist_ok=True)



def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"



def job_paths(job_id: str) -> JobPaths:
    return JobPaths(job_id=job_id)



def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)



def write_job(record: dict[str, Any]) -> None:
    job_id = record["job_id"]
    paths = job_paths(job_id)
    _atomic_write_json(paths.job_json, record)



def read_job(job_id: str) -> dict[str, Any]:
    path = job_paths(job_id).job_json
    if not path.exists():
        raise FileNotFoundError(f"Job {job_id} not found")
    return json.loads(path.read_text(encoding="utf-8"))



def write_lane_result(job_id: str, config: str, payload: dict[str, Any]) -> Path:
    path = job_paths(job_id).lanes_dir / f"{config}.json"
    _atomic_write_json(path, payload)
    return path



def read_lane_result(job_id: str, config: str) -> dict[str, Any]:
    path = job_paths(job_id).lanes_dir / f"{config}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))



def merge_lane_into_job(job_id: str, lane_payload: dict[str, Any]) -> dict[str, Any]:
    record = read_job(job_id)
    config = lane_payload["config"]
    record.setdefault("lanes", {})[config] = lane_payload
    record["updated_at"] = utc_now_iso()
    write_job(record)
    return record



def persist_upload(job_id: str, file_name: str, blob: bytes) -> str:
    model_dir = MODELS_DIR / job_id
    model_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file_name).name or "model.pt"
    if not safe_name.endswith(".pt"):
        safe_name = f"{safe_name}.pt"
    path = model_dir / safe_name
    path.write_bytes(blob)
    return str(path)



def persist_sample_image(job_id: str, file_name: str, blob: bytes) -> str:
    sample_dir = SAMPLES_DIR / job_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file_name).name or "sample.jpg"
    path = sample_dir / safe_name
    path.write_bytes(blob)
    return str(path)



def save_preview(job_id: str, config: str, payload: dict[str, Any]) -> str:
    path = PREVIEWS_DIR / job_id / f"{config}.json"
    _atomic_write_json(path, payload)
    return str(path)



def write_deployment(record: dict[str, Any]) -> None:
    dep_id = record["deployment_id"]
    path = DEPLOYMENTS_DIR / f"{dep_id}.json"
    _atomic_write_json(path, record)



def read_deployment(deployment_id: str) -> dict[str, Any]:
    path = DEPLOYMENTS_DIR / f"{deployment_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Deployment {deployment_id} not found")
    return json.loads(path.read_text(encoding="utf-8"))
