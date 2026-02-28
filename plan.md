# DeployBench — Hackathon Build Plan

## What This Document Is

This is the complete technical specification for building DeployBench at HackIllinois 2026. It is written to be fed directly into Claude Code. Every section describes exactly what to build, how to build it, and what NOT to build. Follow this document as the single source of truth.

---

## One-Liner

DeployBench is a web app where you pick a vision model, we benchmark it across 4 optimization configs on real GPUs simultaneously via Modal, and hand you a comparison dashboard + a deploy-ready Modal script.

"Find the cheapest way to deploy your model — in 60 seconds."

---

## Architecture Overview

```
[Streamlit Frontend]  (hosted as Modal web endpoint)
        |
        | HTTP POST /benchmark {model: "yolov8s"}
        v
[Orchestrator]  (Modal Function, CPU only, no GPU)
        |
        | .spawn() x4 in parallel
        v
[Worker FP32]  [Worker FP16]  [Worker INT8]  [Worker ONNX]
  (T4 GPU)       (T4 GPU)       (T4 GPU)       (T4 GPU)
        |            |              |              |
        | all read from shared Modal Volume (COCO images + labels)
        |
        v
[Orchestrator collects results, computes cost estimates]
        |
        v
[Streamlit renders: table, charts, recommendation, deployment script]
```

---

## Tech Stack

- **Backend:** Python, Modal (Functions, Volume, web_endpoint)
- **Frontend:** Streamlit (single .py file, hosted on Modal)
- **ML Framework:** PyTorch via `ultralytics` library (YOLOv8)
- **Models:** YOLOv8n, YOLOv8s, YOLOv8m (pre-trained COCO checkpoints from ultralytics)
- **Optimization configs:** FP32 (PyTorch), FP16 (PyTorch .half()), INT8 (torch.quantization), ONNX (ultralytics export + onnxruntime-gpu)
- **Evaluation:** pycocotools for mAP computation
- **Charts:** Plotly (built into Streamlit via st.plotly_chart)

---

## Modal Container Image

All workers share one container image. Build it once, cache it.

```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "ultralytics",
        "torch",
        "torchvision",
        "onnxruntime-gpu",
        "pycocotools",
        "opencv-python-headless",
        "numpy",
        "Pillow",
    )
)
```

---

## Modal Volume

One shared Volume mounted read-only by all workers. Contains:

- `coco_val2017/` — 50 validation images (subset of COCO val2017, NOT the full 5K set)
- `coco_annotations/` — instances_val2017.json ground truth labels (filtered to match 50 images)

### Volume Setup Script (run once before demo)

```python
# setup_volume.py — run this once to populate the volume
# Downloads 50 COCO val2017 images and the annotation file
# Uploads them to the Modal Volume

import modal

app = modal.App("deploybench-setup")
volume = modal.Volume.from_name("deploybench-data", create_if_missing=True)

@app.function(volumes={"/data": volume}, timeout=300)
def setup():
    import urllib.request
    import zipfile
    import json
    import os
    import random

    # Download COCO val2017 annotations
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    urllib.request.urlretrieve(ann_url, "/tmp/annotations.zip")
    with zipfile.ZipFile("/tmp/annotations.zip", "r") as z:
        z.extract("annotations/instances_val2017.json", "/tmp")

    # Load annotations and pick 50 random images
    with open("/tmp/annotations/instances_val2017.json") as f:
        coco = json.load(f)

    random.seed(42)
    selected_images = random.sample(coco["images"], 50)
    selected_ids = {img["id"] for img in selected_images}

    # Filter annotations to selected images only
    filtered_anns = [a for a in coco["annotations"] if a["image_id"] in selected_ids]
    filtered_coco = {
        "images": selected_images,
        "annotations": filtered_anns,
        "categories": coco["categories"],
    }

    os.makedirs("/data/coco_annotations", exist_ok=True)
    with open("/data/coco_annotations/instances_val2017.json", "w") as f:
        json.dump(filtered_coco, f)

    # Download the 50 images
    os.makedirs("/data/coco_val2017", exist_ok=True)
    for img in selected_images:
        url = img["coco_url"]
        path = f"/data/coco_val2017/{img['file_name']}"
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)

    volume.commit()
    print(f"Setup complete: {len(selected_images)} images, {len(filtered_anns)} annotations")
```

---

## File Structure

```
deploybench/
├── app.py                  # Modal app definition (image, volume, all functions)
├── orchestrator.py         # Orchestrator function: fans out to workers, collects results
├── workers/
│   ├── base.py             # Shared benchmark logic (warmup, timing loop, mAP calc)
│   ├── fp32_worker.py      # FP32 baseline worker
│   ├── fp16_worker.py      # FP16 half-precision worker
│   ├── int8_worker.py      # INT8 quantized worker
│   └── onnx_worker.py      # ONNX Runtime worker
├── utils/
│   ├── cost_estimator.py   # Converts latency → monthly cost at given request volume
│   ├── metrics.py          # mAP computation using pycocotools
│   └── deploy_script.py   # Generates modal_app.py deployment script string
├── frontend/
│   └── streamlit_app.py    # Streamlit frontend (single file, hosted on Modal)
├── setup_volume.py         # One-time script to populate Modal Volume with COCO data
└── README.md
```

---

## Detailed Component Specs

### 1. Shared Benchmark Logic (`workers/base.py`)

All 4 workers use this same benchmarking procedure. The only difference is how each worker loads the model.

```python
def run_benchmark(model, image_paths, ground_truth, device="cuda"):
    """
    Runs the standard benchmark loop. Returns a dict of metrics.

    Args:
        model: loaded model (PyTorch, ONNX session, etc.)
        image_paths: list of 50 image file paths
        ground_truth: COCO-format annotations for mAP computation
        device: "cuda" or "cpu"

    Returns:
        {
            "avg_latency_ms": float,
            "p95_latency_ms": float,
            "fps": float,
            "peak_memory_mb": float,
            "mAP_50": float,
            "all_latencies": list[float],  # for percentile charts
            "predictions": list[dict],      # for mAP computation
        }
    """
    import torch
    import time

    # Phase 1: Warmup (3 passes, don't measure)
    for i in range(3):
        _ = model(image_paths[0])

    # Phase 2: Reset GPU memory tracking
    torch.cuda.reset_peak_memory_stats()

    # Phase 3: Benchmark loop
    latencies = []
    all_predictions = []

    for img_path in image_paths:
        torch.cuda.synchronize()
        start = time.perf_counter()

        results = model(img_path)

        torch.cuda.synchronize()
        end = time.perf_counter()

        latencies.append((end - start) * 1000)  # ms

        # Collect predictions for mAP
        # Extract boxes, scores, class_ids from results
        # Format as COCO prediction format
        # ... (see metrics.py)

    # Phase 4: Compute metrics
    avg_latency = sum(latencies) / len(latencies)
    sorted_lat = sorted(latencies)
    p95_latency = sorted_lat[int(len(sorted_lat) * 0.95)]
    fps = 1000.0 / avg_latency
    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB

    # Phase 5: Compute mAP using pycocotools
    mAP = compute_mAP(all_predictions, ground_truth)  # from utils/metrics.py

    return {
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "fps": round(fps, 1),
        "peak_memory_mb": round(peak_memory, 1),
        "mAP_50": round(mAP, 4),
    }
```

### 2. Worker Functions

Each worker is a Modal Function with `gpu="T4"`. They mount the shared Volume at `/data`.

**FP32 Worker:**
```python
@app.function(gpu="T4", image=image, volumes={"/data": volume}, timeout=300)
def benchmark_fp32(model_name: str) -> dict:
    from ultralytics import YOLO
    model = YOLO(f"{model_name}.pt")  # loads in FP32 by default
    # ... load image paths and ground truth from /data
    results = run_benchmark(model, image_paths, ground_truth)
    results["config"] = "FP32 (Baseline)"
    return results
```

**FP16 Worker:**
```python
@app.function(gpu="T4", image=image, volumes={"/data": volume}, timeout=300)
def benchmark_fp16(model_name: str) -> dict:
    from ultralytics import YOLO
    model = YOLO(f"{model_name}.pt")
    model.model.half()  # convert all weights to FP16
    # ... run benchmark
    results["config"] = "FP16 (Half Precision)"
    return results
```

**INT8 Worker:**
```python
@app.function(gpu="T4", image=image, volumes={"/data": volume}, timeout=300)
def benchmark_int8(model_name: str) -> dict:
    from ultralytics import YOLO
    import torch
    model = YOLO(f"{model_name}.pt")
    # Dynamic quantization to INT8
    model.model = torch.quantization.quantize_dynamic(
        model.model, {torch.nn.Linear}, dtype=torch.qint8
    )
    # NOTE: If this fails on YOLO architecture, fallback approach:
    # Use ultralytics built-in: model.export(format="ncnn", int8=True)
    # or use model with half() as a "reduced precision" fallback
    # ... run benchmark
    results["config"] = "INT8 (Quantized)"
    return results
```

**ONNX Worker:**
```python
@app.function(gpu="T4", image=image, volumes={"/data": volume}, timeout=300)
def benchmark_onnx(model_name: str) -> dict:
    from ultralytics import YOLO
    import onnxruntime as ort

    # Step 1: Export to ONNX
    model = YOLO(f"{model_name}.pt")
    onnx_path = model.export(format="onnx", half=True, imgsz=640)

    # Step 2: Load with ONNX Runtime CUDA provider
    session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])

    # Step 3: Benchmark using ONNX Runtime inference
    # NOTE: Can't use ultralytics model() call here, need to:
    # - Preprocess images manually (resize, normalize, CHW, batch)
    # - Run session.run()
    # - Postprocess outputs (NMS, box decoding)
    # OR simpler approach: use YOLO(onnx_path) which ultralytics supports natively
    onnx_model = YOLO(onnx_path)
    # ... run benchmark with onnx_model
    results["config"] = "ONNX + FP16"
    return results
```

### 3. Orchestrator (`orchestrator.py`)

```python
@app.function(timeout=600)
def run_all_benchmarks(model_name: str, requests_per_day: int = 10000) -> dict:
    """
    Fans out to all 4 workers in parallel. Collects results.
    Computes cost estimates. Returns everything the frontend needs.
    """
    # Spawn all 4 workers simultaneously
    fp32_call = benchmark_fp32.spawn(model_name)
    fp16_call = benchmark_fp16.spawn(model_name)
    int8_call = benchmark_int8.spawn(model_name)
    onnx_call = benchmark_onnx.spawn(model_name)

    # Wait for all to complete
    results = []
    for call in [fp32_call, fp16_call, int8_call, onnx_call]:
        try:
            result = call.get()
            # Add cost estimate
            result["est_monthly_cost"] = estimate_cost(
                result["avg_latency_ms"], requests_per_day
            )
            results.append(result)
        except Exception as e:
            # If a worker fails, include error but don't crash
            results.append({
                "config": "Unknown",
                "error": str(e),
            })

    # Sort by cost (cheapest first)
    valid_results = [r for r in results if "error" not in r]
    valid_results.sort(key=lambda r: r["est_monthly_cost"])

    # Generate recommendation
    baseline = next(r for r in valid_results if "FP32" in r["config"])
    cheapest = valid_results[0]

    recommendation = {
        "best_config": cheapest["config"],
        "monthly_savings": round(baseline["est_monthly_cost"] - cheapest["est_monthly_cost"], 2),
        "accuracy_tradeoff": round(baseline["mAP_50"] - cheapest["mAP_50"], 4),
        "speedup": round(baseline["avg_latency_ms"] / cheapest["avg_latency_ms"], 2),
    }

    # Generate deployment script for best config
    deploy_script = generate_deploy_script(model_name, cheapest["config"])

    return {
        "model": model_name,
        "results": results,
        "recommendation": recommendation,
        "deploy_script": deploy_script,
    }
```

### 4. Cost Estimator (`utils/cost_estimator.py`)

```python
MODAL_T4_HOURLY_RATE = 0.59  # $/hr for T4 on Modal

def estimate_cost(avg_latency_ms: float, requests_per_day: int) -> float:
    """
    Given average latency and daily request volume, estimate monthly GPU cost.

    Returns: estimated monthly cost in USD
    """
    seconds_per_request = avg_latency_ms / 1000
    total_gpu_seconds_per_day = seconds_per_request * requests_per_day
    gpu_hours_per_day = total_gpu_seconds_per_day / 3600
    monthly_gpu_hours = gpu_hours_per_day * 30
    monthly_cost = monthly_gpu_hours * MODAL_T4_HOURLY_RATE
    return round(monthly_cost, 2)
```

### 5. Deployment Script Generator (`utils/deploy_script.py`)

```python
def generate_deploy_script(model_name: str, config_name: str) -> str:
    """
    Returns a string containing a complete, copy-pasteable modal_app.py
    that deploys the model with the recommended config.
    """

    # Map config name to the relevant model loading code
    load_code = {
        "FP32 (Baseline)": f'''model = YOLO("{model_name}.pt")''',
        "FP16 (Half Precision)": f'''model = YOLO("{model_name}.pt")\n    model.model.half()''',
        "INT8 (Quantized)": f'''model = YOLO("{model_name}.pt")\n    model.model = torch.quantization.quantize_dynamic(model.model, {{torch.nn.Linear}}, dtype=torch.qint8)''',
        "ONNX + FP16": f'''model = YOLO("{model_name}.pt")\n    onnx_path = model.export(format="onnx", half=True)\n    model = YOLO(onnx_path)''',
    }

    return f'''import modal
from ultralytics import YOLO

app = modal.App("my-{model_name}-deployment")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "ultralytics", "torch", "torchvision", "onnxruntime-gpu"
)

@app.function(gpu="T4", image=image)
def predict(image_bytes: bytes) -> dict:
    import io
    from PIL import Image

    {load_code.get(config_name, load_code["FP32 (Baseline)"])}

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
```

### 6. Streamlit Frontend (`frontend/streamlit_app.py`)

```python
import streamlit as st
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="DeployBench", layout="wide")

# ============================================
# HEADER
# ============================================
st.title("DeployBench")
st.caption("Find the cheapest way to deploy your model — in 60 seconds.")

# ============================================
# CONTROLS
# ============================================
col1, col2 = st.columns([2, 1])
with col1:
    model = st.selectbox("Select Model", ["yolov8n", "yolov8s", "yolov8m"])
with col2:
    requests_per_day = st.number_input("Daily requests", value=10000, step=1000)

run = st.button("🚀 Run Benchmark", type="primary", use_container_width=True)

# ============================================
# RESULTS (shown after clicking Run)
# ============================================
if run:
    with st.spinner("Spinning up 4 GPU containers on Modal..."):
        # Call the Modal orchestrator endpoint
        response = requests.post(
            "YOUR_MODAL_ENDPOINT_URL/benchmark",
            json={"model": model, "requests_per_day": requests_per_day},
        )
        data = response.json()

    # --- Results Table ---
    st.subheader("Results")
    results = data["results"]

    # Display as 4 metric cards in a row
    cols = st.columns(len(results))
    for i, r in enumerate(results):
        if "error" in r:
            cols[i].error(f"{r['config']}: Failed")
            continue
        with cols[i]:
            st.metric(r["config"], f"${r['est_monthly_cost']}/mo")
            st.caption(f"{r['avg_latency_ms']}ms · {r['fps']} FPS")
            st.caption(f"mAP: {r['mAP_50']:.1%} · {r['peak_memory_mb']}MB")

    # --- Cost vs Accuracy Chart ---
    st.subheader("Cost vs Accuracy Tradeoff")
    valid = [r for r in results if "error" not in r]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[r["est_monthly_cost"] for r in valid],
        y=[r["mAP_50"] for r in valid],
        mode="markers+text",
        text=[r["config"] for r in valid],
        textposition="top center",
        marker=dict(size=15),
    ))
    fig.update_layout(
        xaxis_title="Estimated Monthly Cost ($)",
        yaxis_title="Accuracy (mAP@50)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Bar chart for cost comparison ---
    st.subheader("Monthly Cost Comparison")
    fig2 = go.Figure(go.Bar(
        x=[r["config"] for r in valid],
        y=[r["est_monthly_cost"] for r in valid],
        marker_color=["#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff"],
    ))
    fig2.update_layout(yaxis_title="Est. Monthly Cost ($)", height=350)
    st.plotly_chart(fig2, use_container_width=True)

    # --- Recommendation ---
    rec = data["recommendation"]
    st.success(
        f"**Recommendation: {rec['best_config']}** — "
        f"Saves ${rec['monthly_savings']}/mo "
        f"({rec['speedup']}x faster, "
        f"{rec['accuracy_tradeoff']:.2%} accuracy tradeoff)"
    )

    # --- Deployment Script ---
    st.subheader("Deploy to Modal")
    st.caption("Copy this script, save as modal_app.py, run `modal deploy modal_app.py`")
    st.code(data["deploy_script"], language="python")
```

---

## What We Are NOT Building (Scope Boundaries)

Do NOT implement any of these. They are out of scope for the hackathon:

- ❌ Custom model upload (user uploading their own .pt file)
- ❌ Modal Sandboxes (mention in pitch as Phase 2 for untrusted uploads)
- ❌ Multiple GPU comparison (T4 vs A10G vs A100) — T4 only
- ❌ LLM benchmarking — vision models only for hackathon
- ❌ User accounts, auth, database, persistence
- ❌ More than 3 model options in the dropdown
- ❌ An "intelligent agent" that picks configs dynamically
- ❌ Batch size optimization or concurrency testing
- ❌ Historical tracking or saved benchmark runs
- ❌ CI/CD integration or GitHub Actions

---

## Known Risks and Fallbacks

| Risk | Mitigation |
|------|-----------|
| `torch.quantization.quantize_dynamic` doesn't work well with YOLO's architecture (conv layers, not linear) | Fallback: use `model.export(format="torchscript")` and benchmark TorchScript instead of INT8. OR use ultralytics built-in `model.export(format="engine")` for TensorRT if available. OR simply label the config as "Dynamic INT8" and note in results that conv-heavy models benefit less from dynamic quantization. |
| ONNX export fails on specific YOLOv8 variant | ultralytics has well-tested ONNX export via `model.export(format="onnx")`. This is unlikely to fail. If it does, skip ONNX worker and show 3 results instead of 4. |
| Modal cold starts add 10-20s latency on first run | Use `keep_warm=1` on at least one worker. Accept that first benchmark takes ~90s. Subsequent runs use cached containers and complete in ~30-60s. Pre-warm containers before the demo. |
| mAP computation is slow or produces unexpected results | Use only 50 images (not full COCO 5K). If pycocotools is problematic, fallback to simple IoU-based accuracy: compute average IoU between predicted and ground truth boxes. Less rigorous but still shows relative accuracy difference between configs. |
| Credits run out | Budget: 4 workers × T4 × ~2 min each = ~8 min GPU time per benchmark run. At $0.59/hr = $0.08/run. We can do 3000+ runs on $250 credits. Not a risk. |
| Streamlit is too slow or limited for the UI we want | Streamlit is sufficient for this scope. If we hit layout limitations, Gradio is the alternative (also one-file Python, also hosted on Modal). |

---

## Build Order (Priority Sequence)

This is the order to build things. Do NOT skip ahead. Each step should be working before moving to the next.

### Phase 1: Foundation (Hours 0-6)
1. Create Modal app with the container image definition
2. Write and run `setup_volume.py` to populate COCO data
3. Build ONE worker (FP32) end-to-end: loads model, runs 50 images, returns timing + accuracy
4. Test it: `modal run app.py::benchmark_fp32 --model-name yolov8s`
5. Verify: you get back a dict with avg_latency_ms, fps, mAP_50, peak_memory_mb

### Phase 2: All Workers (Hours 6-12)
6. Duplicate FP32 worker → FP16 worker (add `.half()`)
7. Duplicate FP32 worker → INT8 worker (add quantization)
8. Duplicate FP32 worker → ONNX worker (add export + ONNX Runtime)
9. Test each independently. If INT8 fails, see fallback above.
10. Write orchestrator that spawns all 4, collects results

### Phase 3: Cost + Recommendation (Hours 12-15)
11. Add cost estimator logic
12. Add recommendation logic (pick best cost-accuracy tradeoff)
13. Add deployment script generator
14. Test orchestrator end-to-end: one function call returns everything the frontend needs

### Phase 4: Frontend (Hours 15-22)
15. Build Streamlit app with hardcoded mock data first (get the layout right)
16. Connect to real Modal endpoint
17. Add metric cards, cost bar chart, accuracy scatter plot
18. Add recommendation display
19. Add deployment script code block with copy button
20. Deploy Streamlit as Modal web endpoint

### Phase 5: Polish (Hours 22-30)
21. Pre-warm containers before demo
22. Add error handling (worker failure doesn't crash the app)
23. Test with all 3 model sizes (yolov8n, yolov8s, yolov8m)
24. Add a sample image upload panel (STRETCH GOAL) — user picks an image, all 4 configs draw bounding boxes on it side by side
25. Record 3-minute demo video
26. Prepare pitch deck
27. Write README

---

## Environment Setup Commands

```bash
# Install Modal CLI
pip install modal
modal token new

# Set up the Modal app and volume
modal run setup_volume.py

# Test a single worker
modal run app.py::benchmark_fp32 --model-name yolov8s

# Test the full orchestrator
modal run app.py::run_all_benchmarks --model-name yolov8s

# Deploy the Streamlit frontend
modal deploy app.py

# Pre-warm containers before demo
modal run app.py::warmup
```

---

## Key Modal Patterns to Use

### Spawning parallel functions:
```python
call = my_function.spawn(args)  # non-blocking
result = call.get()              # blocks until done
```

### Web endpoint for the orchestrator:
```python
@app.function()
@modal.web_endpoint(method="POST")
def benchmark(item: dict):
    return run_all_benchmarks.remote(item["model"], item.get("requests_per_day", 10000))
```

### Hosting Streamlit on Modal:
```python
@app.function(image=streamlit_image, allow_concurrent_inputs=10)
@modal.asgi_app()
def streamlit():
    from starlette.applications import Starlette
    from starlette.routing import Mount
    # ... or use modal's built-in Streamlit serving pattern
```

Note: Check Modal docs for the latest pattern for serving Streamlit. An alternative is to run Streamlit locally during development and only deploy the backend API on Modal.

---

## Final Checklist Before Submission

- [ ] All 4 workers return valid results for yolov8n, yolov8s, yolov8m
- [ ] Orchestrator fans out in parallel (verify via Modal dashboard that 4 containers spin up simultaneously)
- [ ] Cost estimates are reasonable (sanity check: FP32 should be most expensive, INT8 or ONNX cheapest)
- [ ] mAP numbers are reasonable (should be 85-95% range for COCO, not 0% or 100%)
- [ ] Recommendation picks the best cost-accuracy tradeoff (not just the cheapest)
- [ ] Deployment script is valid Python that would actually work if copy-pasted
- [ ] Frontend displays all results without errors
- [ ] 3-minute demo video is recorded
- [ ] README explains what DeployBench is and how to run it
- [ ] Pre-warm containers so demo doesn't have 60s cold start