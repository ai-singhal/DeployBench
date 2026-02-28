"""Job lifecycle helpers for DeployBench."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from backend.recommendation import build_recommendation, compute_agreement
from backend.storage import utc_now_iso

LANE_CONFIGS = ["FP32", "FP16", "INT8", "ONNX_FP16"]
TERMINAL_STATUSES = {"done", "failed", "unsupported"}



def create_job_record(
    *,
    job_id: str,
    mode: str,
    requests_per_day: int,
    task_hint: str | None,
    model_name: str | None,
    model_ref: str | None,
    sample_image_ref: str | None,
) -> dict[str, Any]:
    lanes = {
        cfg: {
            "config": cfg,
            "status": "queued",
            "avg_latency_ms": None,
            "p50_latency_ms": None,
            "p95_latency_ms": None,
            "fps": None,
            "peak_memory_mb": None,
            "gpu_monitor_source": None,
            "gpu_monitor_sample_count": 0,
            "gpu_util_avg_pct": None,
            "gpu_util_p95_pct": None,
            "gpu_util_peak_pct": None,
            "gpu_mem_util_avg_pct": None,
            "gpu_mem_used_peak_mb": None,
            "gpu_mem_used_peak_pct": None,
            "gpu_power_avg_w": None,
            "gpu_temp_peak_c": None,
            "gpu_monitor_errors": [],
            "quality_metric": None,
            "quality_metric_name": None,
            "quality_vs_fp32_delta": None,
            "est_monthly_cost": None,
            "warnings": [],
            "error": None,
        }
        for cfg in LANE_CONFIGS
    }

    now = utc_now_iso()
    return {
        "job_id": job_id,
        "mode": mode,
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "requests_per_day": int(requests_per_day),
        "task": task_hint,
        "model_name": model_name,
        "model_ref": model_ref,
        "sample_image_ref": sample_image_ref,
        "lanes": lanes,
        "recommendation": None,
        "deployment": None,
        "errors": [],
    }



def merge_lane(record: dict[str, Any], lane_payload: dict[str, Any]) -> dict[str, Any]:
    cfg = lane_payload["config"]
    if cfg not in record["lanes"]:
        raise KeyError(f"Unknown lane config {cfg}")
    merged = deepcopy(record)
    merged["lanes"][cfg].update(lane_payload)
    merged["updated_at"] = utc_now_iso()
    return merged



def _all_lanes_terminal(record: dict[str, Any]) -> bool:
    return all(lane.get("status") in TERMINAL_STATUSES for lane in record["lanes"].values())



def _quality_name_for_task(task: str | None) -> str:
    if task == "classify":
        return "Top-1 Agreement"
    if task == "segment":
        return "Mask/Class Agreement"
    return "mAP50" if task == "detect" else "Agreement"



def finalize_job(record: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(record)
    task = merged.get("task") or "detect"

    baseline = merged["lanes"].get("FP32")
    if baseline and baseline.get("status") == "done":
        base_sig = baseline.get("_quality_signature")
        if baseline.get("quality_metric") is None:
            baseline["quality_metric"] = 1.0
            baseline["quality_metric_name"] = _quality_name_for_task(task)
            baseline["quality_vs_fp32_delta"] = 0.0

        for cfg, lane in merged["lanes"].items():
            if cfg == "FP32" or lane.get("status") != "done":
                continue
            if lane.get("quality_metric") is not None:
                # Already has true metric (e.g. gallery detection); keep it.
                continue
            agreement = compute_agreement(base_sig or {}, lane.get("_quality_signature") or {}, task)
            lane["quality_metric"] = agreement
            lane["quality_metric_name"] = _quality_name_for_task(task)
            lane["quality_vs_fp32_delta"] = round(1.0 - agreement, 4)

    merged["recommendation"] = build_recommendation(merged["lanes"], task)

    if _all_lanes_terminal(merged):
        if all(l.get("status") == "unsupported" for l in merged["lanes"].values()):
            merged["status"] = "incompatible"
        elif any(l.get("status") == "done" for l in merged["lanes"].values()):
            merged["status"] = "completed"
        else:
            merged["status"] = "failed"
    else:
        merged["status"] = "running"

    merged["updated_at"] = utc_now_iso()
    return merged



def public_job(record: dict[str, Any]) -> dict[str, Any]:
    safe = deepcopy(record)
    for lane in safe.get("lanes", {}).values():
        lane.pop("_quality_signature", None)
    return safe
