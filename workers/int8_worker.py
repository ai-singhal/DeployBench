"""
INT8 dynamic quantization worker.

torch.quantization.quantize_dynamic targets Linear layers only. YOLO is
conv-heavy, so the quantization may apply to very few layers or fail entirely
on CUDA. Both failure modes are caught and the worker falls back to FP32 with
a warning in the results dict.
"""

import logging

import torch
from ultralytics import YOLO

from workers.base import load_data, run_benchmark


def benchmark_int8(model_name: str) -> dict:
    model = YOLO(f"{model_name}.pt")
    warning = None

    try:
        model.model = torch.quantization.quantize_dynamic(
            model.model, {torch.nn.Linear}, dtype=torch.qint8
        )
    except Exception as e:
        warning = (
            f"Dynamic INT8 quantization failed ({e}). "
            "Conv-heavy YOLO architectures see minimal benefit from dynamic INT8 "
            "quantization, which only targets Linear layers. "
            "Running baseline FP32 precision instead."
        )
        logging.warning(warning)

    image_paths, image_ids, ground_truth, coco_cat_id_map = load_data()

    try:
        results = run_benchmark(model, image_paths, image_ids, ground_truth, coco_cat_id_map)
    except Exception as e:
        # Quantized ops are CPU-only; reload a clean FP32 model and retry on GPU
        if warning is None:
            warning = (
                f"INT8 quantized model inference failed on GPU ({e}). "
                "Dynamic INT8 quantization is CPU-only and incompatible with CUDA. "
                "Running baseline FP32 precision instead."
            )
            logging.warning(warning)
        model = YOLO(f"{model_name}.pt")
        results = run_benchmark(model, image_paths, image_ids, ground_truth, coco_cat_id_map)

    results["config"] = "INT8 (Quantized)"
    if warning:
        results["warning"] = warning
    return results
