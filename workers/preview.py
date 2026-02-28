"""Preview and deployed inference helpers."""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from PIL import Image

from backend.storage import read_deployment, read_job
from workers.pipeline import LaneUnsupportedError, _build_model_for_config



def _image_from_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")



def _encode_plot_base64(yolo_result: Any) -> str:
    plotted = yolo_result.plot()
    if plotted is None:
        return ""
    arr = np.asarray(plotted)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr[:, :, ::-1]  # BGR -> RGB
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")



def _normalize_output(task: str, result: Any) -> dict[str, Any]:
    if task == "classify":
        probs = getattr(result, "probs", None)
        if probs is None:
            return {"top1": None, "top5": []}
        top1 = int(getattr(probs, "top1", -1))
        names = getattr(result, "names", {}) or {}
        top5_ids = list(getattr(probs, "top5", []) or [])
        top5_conf = list(getattr(probs, "top5conf", []) or [])
        top5 = []
        for idx, conf in zip(top5_ids, top5_conf):
            top5.append({
                "class_id": int(idx),
                "class_name": names.get(int(idx), str(idx)),
                "confidence": float(conf),
            })
        return {"top1": top1, "top1_name": names.get(top1, str(top1)), "top5": top5}

    boxes = getattr(result, "boxes", None)
    names = getattr(result, "names", {}) or {}
    detections = []
    if boxes is not None:
        for box in boxes:
            cls = int(box.cls[0])
            detections.append(
                {
                    "class_id": cls,
                    "class_name": names.get(cls, str(cls)),
                    "confidence": float(box.conf[0]),
                    "bbox_xyxy": [float(x) for x in box.xyxy[0].tolist()],
                }
            )

    payload: dict[str, Any] = {"detections": detections}

    if task == "segment":
        masks = getattr(result, "masks", None)
        payload["mask_count"] = len(masks) if masks is not None else 0

    return payload



def _infer_single(model_ref: str, config: str, task: str, image_bytes: bytes) -> dict[str, Any]:
    model, infer_kwargs, warnings = _build_model_for_config(model_ref, config, task)
    image = _image_from_bytes(image_bytes)
    result = model(image, **infer_kwargs)[0]
    return {
        "config": config,
        "task": task,
        "warnings": warnings,
        "outputs": _normalize_output(task, result),
        "annotated_image_b64": _encode_plot_base64(result),
    }



def run_preview(job_id: str, image_bytes: bytes, config: str | None = None) -> dict[str, Any]:
    record = read_job(job_id)
    task = record.get("task") or "detect"
    model_ref = record.get("model_ref")
    if record.get("mode") == "gallery":
        model_ref = f"{record.get('model_name')}.pt"

    if not model_ref:
        raise RuntimeError("Model reference missing from job")

    if config:
        target_configs = [config]
    else:
        target_configs = [
            cfg
            for cfg, lane in record.get("lanes", {}).items()
            if lane.get("status") == "done"
        ]

    outputs = {}
    for cfg in target_configs:
        try:
            outputs[cfg] = _infer_single(model_ref, cfg, task, image_bytes)
        except LaneUnsupportedError as exc:
            outputs[cfg] = {"config": cfg, "status": "unsupported", "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            outputs[cfg] = {"config": cfg, "status": "failed", "error": str(exc)}

    return {
        "job_id": job_id,
        "task": task,
        "results": outputs,
    }



def run_deployment_infer(deployment_id: str, image_bytes: bytes) -> dict[str, Any]:
    dep = read_deployment(deployment_id)
    job = read_job(dep["job_id"])
    task = dep.get("task") or job.get("task") or "detect"

    model_ref = dep.get("artifact_paths", {}).get("model_ref") or job.get("model_ref")
    if job.get("mode") == "gallery":
        model_ref = f"{job.get('model_name')}.pt"

    config = dep.get("config", "FP32")
    result = _infer_single(model_ref, config, task, image_bytes)

    return {
        "deployment_id": deployment_id,
        "config": config,
        "task": task,
        "outputs": result["outputs"],
    }
