"""Shared benchmark pipeline for DeployBench lanes."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from backend.costs import estimate_monthly_cost
from utils.metrics import compute_mAP, format_predictions
from workers.gpu_monitor import GPUMonitor

DATA_DIR = "/data"
IMAGE_DIR = os.path.join(DATA_DIR, "coco_val2017")
ANN_FILE = os.path.join(DATA_DIR, "coco_annotations", "instances_val2017.json")


@dataclass
class LaneContext:
    config: str
    mode: str  # gallery | upload
    model_ref: str
    task: str
    requests_per_day: int


class LaneUnsupportedError(RuntimeError):
    """Raised when a lane is not safely supported for a given model/task."""



def _load_eval_images(max_images: int = 20) -> tuple[list[str], list[int], dict, dict]:
    with open(ANN_FILE, encoding="utf-8") as f:
        gt = json.load(f)

    images = gt.get("images", [])[:max_images]
    image_paths = [os.path.join(IMAGE_DIR, i["file_name"]) for i in images]
    image_ids = [i["id"] for i in images]
    cat_map = {cat["name"]: cat["id"] for cat in gt.get("categories", [])}

    selected_ids = set(image_ids)
    filtered_gt = {
        "images": images,
        "annotations": [a for a in gt.get("annotations", []) if a.get("image_id") in selected_ids],
        "categories": gt.get("categories", []),
    }
    return image_paths, image_ids, filtered_gt, cat_map



def _predict_kwargs(config: str) -> dict[str, Any]:
    if config == "FP16":
        return {"half": True, "device": "cuda", "verbose": False}
    return {"verbose": False}



def _build_model_for_config(model_ref: str, config: str, task: str) -> tuple[Any, dict[str, Any], list[str]]:
    warnings: list[str] = []

    model = YOLO(model_ref)

    if config == "FP16":
        if not torch.cuda.is_available():
            raise LaneUnsupportedError("FP16 requires CUDA but GPU is not available.")
        return model, _predict_kwargs(config), warnings

    if config == "INT8":
        if task in {"detect", "segment"}:
            raise LaneUnsupportedError(
                "INT8 dynamic quantization is unsafe for conv-heavy detect/segment models in this v1 pipeline."
            )
        try:
            model.model = torch.quantization.quantize_dynamic(
                model.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            warnings.append("INT8 uses torch dynamic quantization (Linear layers only).")
        except Exception as exc:  # noqa: BLE001
            raise LaneUnsupportedError(f"INT8 quantization failed: {exc}") from exc
        return model, _predict_kwargs(config), warnings

    if config == "ONNX_FP16":
        try:
            onnx_path = str(model.export(format="onnx", half=True, imgsz=640, verbose=False))
            onnx_model = YOLO(onnx_path)
            return onnx_model, _predict_kwargs(config), warnings
        except Exception as exc:  # noqa: BLE001
            raise LaneUnsupportedError(f"ONNX FP16 export/load failed: {exc}") from exc

    # FP32 baseline
    return model, _predict_kwargs(config), warnings



def _summarize_result(task: str, yolo_result: Any) -> dict[str, Any]:
    if task == "classify":
        probs = getattr(yolo_result, "probs", None)
        if probs is None:
            return {"top1": None, "classes": []}
        top1 = int(getattr(probs, "top1", -1))
        return {"top1": top1, "classes": [top1] if top1 >= 0 else []}

    boxes = getattr(yolo_result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return {"count": 0, "classes": []}
    cls_ids = boxes.cls.cpu().numpy().astype(int).tolist()
    return {"count": len(cls_ids), "classes": sorted(set(cls_ids))}



def run_lane(ctx: LaneContext) -> dict[str, Any]:
    image_paths, image_ids, ground_truth, coco_cat_id_map = _load_eval_images(max_images=20)
    model, infer_kwargs, warnings = _build_model_for_config(ctx.model_ref, ctx.config, ctx.task)

    # Warmup (unmeasured)
    for _ in range(5):
        model(image_paths[0], **infer_kwargs)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    latencies: list[float] = []
    predictions = []
    signatures: dict[str, Any] = {"images": {}}
    gpu_monitor = GPUMonitor(poll_interval_sec=0.2)
    gpu_monitor.start()

    try:
        for img_path, image_id in zip(image_paths, image_ids):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = model(img_path, **infer_kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)

            first = result[0]
            signatures["images"][str(image_id)] = _summarize_result(ctx.task, first)

            if ctx.task == "detect":
                predictions.extend(format_predictions(result, image_id, coco_cat_id_map))
    finally:
        gpu_monitor.stop()

    lat_sorted = sorted(latencies)
    n = len(lat_sorted)
    avg_latency = sum(lat_sorted) / n
    p50_latency = lat_sorted[n // 2]
    p95_latency = lat_sorted[min(n - 1, int(n * 0.95))]
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    peak_mem = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if torch.cuda.is_available()
        else 0.0
    )

    quality_metric = None
    quality_metric_name = None
    if ctx.mode == "gallery" and ctx.task == "detect":
        quality_metric = compute_mAP(predictions, ground_truth)
        quality_metric_name = "mAP50"

    result = {
        "config": ctx.config,
        "status": "done",
        "avg_latency_ms": round(avg_latency, 2),
        "p50_latency_ms": round(p50_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "fps": round(fps, 2),
        "peak_memory_mb": round(peak_mem, 2),
        "quality_metric": round(quality_metric, 4) if quality_metric is not None else None,
        "quality_metric_name": quality_metric_name,
        "quality_vs_fp32_delta": None,
        "warnings": warnings,
        "est_monthly_cost": estimate_monthly_cost(avg_latency, ctx.requests_per_day),
        "_quality_signature": signatures,
    }
    result.update(gpu_monitor.summary())
    return result



def normalize_lane_error(config: str, err: Exception, status: str = "failed") -> dict[str, Any]:
    return {
        "config": config,
        "status": status,
        "error": str(err),
        "warnings": [],
        "quality_metric": None,
        "quality_metric_name": None,
        "quality_vs_fp32_delta": None,
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
    }
