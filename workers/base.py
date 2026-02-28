"""
Shared benchmark logic for all DeployBench workers.

All 4 workers (FP32, FP16, INT8, ONNX) call run_benchmark() from here.
The only thing that differs per worker is how the model is loaded.
"""

import json
import os
import time

DATA_DIR = "/data"
IMAGE_DIR = os.path.join(DATA_DIR, "coco_val2017")
ANN_FILE = os.path.join(DATA_DIR, "coco_annotations", "instances_val2017.json")


def load_data() -> tuple:
    """
    Load COCO validation image paths and ground truth annotations from the Modal Volume.

    Returns:
        image_paths: list of absolute file paths (one per image)
        image_ids: list of COCO image IDs matching image_paths
        ground_truth: full COCO annotation dict
        coco_cat_id_map: dict mapping class_name -> COCO category_id
    """
    with open(ANN_FILE) as f:
        ground_truth = json.load(f)

    image_paths = []
    image_ids = []
    for img in ground_truth["images"]:
        path = os.path.join(IMAGE_DIR, img["file_name"])
        image_paths.append(path)
        image_ids.append(img["id"])

    # COCO category IDs are not 0-indexed; build name -> id map for prediction formatting
    coco_cat_id_map = {cat["name"]: cat["id"] for cat in ground_truth["categories"]}

    return image_paths, image_ids, ground_truth, coco_cat_id_map


def run_benchmark(model, image_paths: list, image_ids: list,
                  ground_truth: dict, coco_cat_id_map: dict,
                  device: str = "cuda") -> dict:
    """
    Standard benchmark loop shared by all workers.

    Phases:
      1. Warmup — 3 unmeasured passes on the first image
      2. Reset GPU peak memory counter
      3. Timed inference over all images with synchronization barriers
      4. Compute latency statistics and peak memory
      5. Compute mAP@50 via pycocotools

    Args:
        model: loaded model compatible with ultralytics YOLO predict API
        image_paths: list of image file paths
        image_ids: list of COCO image IDs (parallel to image_paths)
        ground_truth: COCO annotation dict
        coco_cat_id_map: class_name -> COCO category_id mapping
        device: "cuda" or "cpu"

    Returns:
        dict with keys:
            avg_latency_ms, p95_latency_ms, fps,
            peak_memory_mb, mAP_50
    """
    import torch
    from utils.metrics import format_predictions, compute_mAP

    # Phase 1: Warmup (3 passes, not measured)
    for _ in range(3):
        model(image_paths[0], verbose=False)

    # Phase 2: Reset GPU memory tracking
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Phase 3: Timed inference loop
    latencies = []
    all_predictions = []

    use_cuda = device == "cuda" and torch.cuda.is_available()

    for img_path, img_id in zip(image_paths, image_ids):
        if use_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()

        results = model(img_path, verbose=False)

        if use_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()

        latencies.append((end - start) * 1000)  # convert to ms

        preds = format_predictions(results, img_id, coco_cat_id_map)
        all_predictions.extend(preds)

    # Phase 4: Compute latency statistics
    n = len(latencies)
    avg_latency = sum(latencies) / n
    sorted_lat = sorted(latencies)
    p95_latency = sorted_lat[int(n * 0.95)]
    fps = 1000.0 / avg_latency

    if use_cuda:
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    else:
        peak_memory = 0.0

    # Phase 5: mAP@50 via pycocotools
    mAP = compute_mAP(all_predictions, ground_truth)

    return {
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "fps": round(fps, 1),
        "peak_memory_mb": round(peak_memory, 1),
        "mAP_50": round(mAP, 4),
    }
