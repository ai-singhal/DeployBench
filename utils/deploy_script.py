"""
Generates a complete, copy-pasteable Modal deployment script for the recommended config.
"""

# Maps benchmark config names to the model-loading code block used inside the predict function.
_LOAD_CODE: dict[str, str] = {
    "FP32 (Baseline)": 'model = YOLO("{model_name}.pt")',
    "FP16 (Half Precision)": (
        'model = YOLO("{model_name}.pt")\n'
        "    model.model.half()"
    ),
    "INT8 (Quantized)": (
        'model = YOLO("{model_name}.pt")\n'
        "    model.model = torch.quantization.quantize_dynamic(\n"
        "        model.model, {{torch.nn.Linear}}, dtype=torch.qint8\n"
        "    )"
    ),
    "ONNX + FP16": (
        'model = YOLO("{model_name}.pt")\n'
        "    onnx_path = model.export(format=\"onnx\", half=True)\n"
        '    model = YOLO(onnx_path)'
    ),
}

_SCRIPT_TEMPLATE = '''\
import modal
from ultralytics import YOLO
{extra_imports}
app = modal.App("my-{model_name}-deployment")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "ultralytics", "torch", "torchvision", "onnxruntime-gpu"
)


@app.function(gpu="T4", image=image)
def predict(image_bytes: bytes) -> dict:
    import io
    from PIL import Image

    {load_code}

    img = Image.open(io.BytesIO(image_bytes))
    results = model(img)

    detections = []
    for box in results[0].boxes:
        detections.append({{
            "class": results[0].names[int(box.cls[0])],
            "confidence": float(box.conf[0]),
            "bbox": box.xyxy[0].tolist(),
        }})
    return {{"detections": detections}}


@app.function(image=image)
@modal.web_endpoint(method="POST")
def api(image_bytes: bytes):
    return predict.remote(image_bytes)
'''


def generate_deploy_script(model_name: str, config_name: str) -> str:
    """
    Return a complete, valid modal_app.py as a string for the given model and config.

    Args:
        model_name: e.g. "yolov8s"
        config_name: one of the benchmark config names (e.g. "FP32 (Baseline)")

    Returns:
        Python source code string ready to save as modal_app.py.
    """
    if not model_name or not isinstance(model_name, str):
        raise ValueError(f"model_name must be a non-empty string, got {model_name!r}")

    known = config_name in _LOAD_CODE
    load_code_template = _LOAD_CODE.get(config_name, _LOAD_CODE["FP32 (Baseline)"])
    load_code = load_code_template.format(model_name=model_name)

    if not known:
        load_code = (
            f"# NOTE: config '{config_name}' is unknown — defaulting to FP32 baseline.\n"
            f"    {load_code}"
        )

    # INT8 config needs torch imported at module level in the generated script
    extra_imports = "\nimport torch\n" if "INT8" in config_name else ""

    return _SCRIPT_TEMPLATE.format(
        model_name=model_name,
        load_code=load_code,
        extra_imports=extra_imports,
    )
