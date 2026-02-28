# DeployBench

DeployBench benchmarks vision model optimization configs in parallel on Modal and recommends a production deployment path.

## What it now supports

- Upload `.pt` checkpoints (max 500MB) or use gallery models.
- Job-based API with polling (`/v1/jobs`, `/v1/jobs/{id}`).
- 4-lane optimization race: `FP32`, `FP16`, `INT8`, `ONNX_FP16`.
- Upload-mode lanes execute in Modal Sandboxes (gallery lanes use Functions).
- Live recommendation with cost/speed/quality tradeoff.
- Per-lane GPU telemetry: utilization avg/p95/peak, memory usage, and power draw.
- Preview inference per config.
- One-click deployment to shared authenticated inference gateway.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Deploy backend on Modal

```bash
source .venv/bin/activate
modal deploy app.py
```

Main API endpoint is the `api_app` ASGI function URL, typically:

```text
https://<workspace>--deploybench-api-app.modal.run
```

## Run frontend

```bash
source .venv/bin/activate
streamlit run frontend/streamlit_app.py
```

In the sidebar, set the base API URL to your deployed Modal URL.

## API routes

- `POST /v1/jobs`
- `GET /v1/jobs/{job_id}`
- `POST /v1/jobs/{job_id}/preview`
- `POST /v1/jobs/{job_id}/deploy`
- `POST /v1/infer/{deployment_id}`

## Data volumes

- `deploybench-data`: eval dataset for benchmarking.
- `deploybench-artifacts`: jobs, uploaded models, lane outputs, deployments.

Use `setup_volume.py` once to populate `deploybench-data`.
