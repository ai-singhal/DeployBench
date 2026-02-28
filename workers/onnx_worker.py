"""
ONNX + FP16 worker.

Exports the PyTorch model to ONNX (half precision) then loads it back via
ultralytics YOLO, which uses onnxruntime-gpu under the hood. If export or
inference fails, returns an error dict instead of raising.
"""

from ultralytics import YOLO

from workers.base import load_data, run_benchmark


def benchmark_onnx(model_name: str) -> dict:
    try:
        model = YOLO(f"{model_name}.pt")
        onnx_path = str(model.export(format="onnx", half=True, imgsz=640, verbose=False))
        onnx_model = YOLO(onnx_path)
        image_paths, image_ids, ground_truth, coco_cat_id_map = load_data()
        results = run_benchmark(onnx_model, image_paths, image_ids, ground_truth, coco_cat_id_map)
        results["config"] = "ONNX + FP16"
        return results
    except Exception as e:
        return {"config": "ONNX + FP16", "error": str(e)}
