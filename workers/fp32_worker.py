"""FP32 baseline worker — no optimization, full float32 precision."""

from ultralytics import YOLO

from workers.base import load_data, run_benchmark


def benchmark_fp32(model_name: str) -> dict:
    model = YOLO(f"{model_name}.pt")
    image_paths, image_ids, ground_truth, coco_cat_id_map = load_data()
    results = run_benchmark(model, image_paths, image_ids, ground_truth, coco_cat_id_map)
    results["config"] = "FP32 (Baseline)"
    return results
