"""FP16 half-precision worker."""

from ultralytics import YOLO

from workers.base import load_data, run_benchmark


def benchmark_fp16(model_name: str) -> dict:
    model = YOLO(f"{model_name}.pt")
    image_paths, image_ids, ground_truth, coco_cat_id_map = load_data()
    results = run_benchmark(
        model,
        image_paths,
        image_ids,
        ground_truth,
        coco_cat_id_map,
        predict_kwargs={"half": True, "device": "cuda"},
    )
    results["config"] = "FP16 (Half Precision)"
    return results
