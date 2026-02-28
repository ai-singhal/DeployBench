"""FP16 half-precision worker — converts all weights to float16."""

from ultralytics import YOLO

from workers.base import load_data, run_benchmark


def benchmark_fp16(model_name: str) -> dict:
    model = YOLO(f"{model_name}.pt")
    model.model.half()  # convert all weights to FP16
    image_paths, image_ids, ground_truth, coco_cat_id_map = load_data()
    results = run_benchmark(model, image_paths, image_ids, ground_truth, coco_cat_id_map)
    results["config"] = "FP16 (Half Precision)"
    return results
