"""
app.py — Single Modal entry point for DeployBench.

Defines the container image, shared volume, all GPU worker functions, the
orchestrator function, and the HTTP web endpoint.
"""

import modal

app = modal.App("deploybench")

# Local modules needed inside Modal containers.
_SOURCE_MODULES = ("workers", "utils", "orchestrator")

# ---------------------------------------------------------------------------
# Container image — shared by all GPU workers
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
    )
    .pip_install(
        "fastapi[standard]",
        "ultralytics",
        "torch",
        "torchvision",
        "onnxruntime-gpu",
        "pycocotools",
        "opencv-python-headless",
        "numpy",
        "Pillow",
    )
    .add_local_python_source(*_SOURCE_MODULES)
)

# Image for HTTP endpoint functions (must include FastAPI explicitly).
api_image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi[standard]")

# Lightweight CPU image for the orchestrator logic.
orchestrator_image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_python_source("orchestrator", "utils")
)

# ---------------------------------------------------------------------------
# Shared volume — pre-populated by setup_volume.py (run once before demo)
# ---------------------------------------------------------------------------
volume = modal.Volume.from_name("deploybench-data", create_if_missing=True)

# ---------------------------------------------------------------------------
# Shared kwargs for all GPU worker functions (DRY — avoids repeating 4×)
# ---------------------------------------------------------------------------
_WORKER_KWARGS = dict(
    gpu="T4",
    image=image,
    volumes={"/data": volume},
    timeout=300,
)

# ---------------------------------------------------------------------------
# Worker functions — each delegates to its pure-Python module
# ---------------------------------------------------------------------------

@app.function(**_WORKER_KWARGS)
def benchmark_fp32(model_name: str) -> dict:
    from workers.fp32_worker import benchmark_fp32 as _run
    return _run(model_name)


@app.function(**_WORKER_KWARGS)
def benchmark_fp16(model_name: str) -> dict:
    from workers.fp16_worker import benchmark_fp16 as _run
    return _run(model_name)


@app.function(**_WORKER_KWARGS)
def benchmark_int8(model_name: str) -> dict:
    from workers.int8_worker import benchmark_int8 as _run
    return _run(model_name)


@app.function(**_WORKER_KWARGS)
def benchmark_onnx(model_name: str) -> dict:
    from workers.onnx_worker import benchmark_onnx as _run
    return _run(model_name)


# ---------------------------------------------------------------------------
# Orchestrator Modal function — CPU only; fans out to GPU workers in parallel
# ---------------------------------------------------------------------------

@app.function(image=orchestrator_image, timeout=600)
def run_all_benchmarks(model_name: str, requests_per_day: int = 10000) -> dict:
    """
    Spawn all 4 GPU workers simultaneously, collect results, compute cost
    estimates and a recommendation, and return everything the frontend needs.
    """
    from orchestrator import run_all_benchmarks as orchestrate

    calls_dict = {
        "fp32": benchmark_fp32.spawn(model_name),
        "fp16": benchmark_fp16.spawn(model_name),
        "int8": benchmark_int8.spawn(model_name),
        "onnx": benchmark_onnx.spawn(model_name),
    }
    return orchestrate(model_name, requests_per_day, calls_dict)


# ---------------------------------------------------------------------------
# HTTP web endpoint — accepts {"model": str, "requests_per_day": int}
# ---------------------------------------------------------------------------

@app.function(image=api_image, timeout=660)
@modal.fastapi_endpoint(method="POST")
def benchmark(item: dict) -> dict:
    """
    POST /benchmark
    Body: {"model": "yolov8s", "requests_per_day": 10000}
    """
    model = item.get("model", "yolov8s")
    requests_per_day = int(item.get("requests_per_day", 10000))
    return run_all_benchmarks.remote(model, requests_per_day)
