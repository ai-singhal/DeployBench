# DeployBench

Benchmark multiple deployment configs (FP32, FP16, INT8, ONNX) and compare cost/latency/accuracy in Streamlit.

## 1) Use Python 3.11 (local)

This project should run on Python 3.11 to avoid type-hint/runtime compatibility issues.

```bash
rm -rf .venv
/Users/manningwu/miniconda3/envs/chatbot/bin/python -m venv .venv
source .venv/bin/activate
python --version
pip install -r requirements.txt
```

Expected version:

```bash
Python 3.11.x
```

## 2) Deploy backend on Modal

From the repo root:

```bash
source .venv/bin/activate
modal deploy app.py
```

The backend endpoint for function `benchmark` is typically this format:

```text
https://<workspace>--deploybench-benchmark.modal.run
```

For your workspace, use:

```text
https://manningwu07--deploybench-benchmark.modal.run
```

## 3) Run frontend and connect to backend

```bash
source .venv/bin/activate
streamlit run frontend/streamlit_app.py
```

In the Streamlit sidebar, set **Modal endpoint URL** to:

```text
https://manningwu07--deploybench-benchmark.modal.run
```

You can also use the explicit benchmark path if needed:

```text
https://manningwu07--deploybench-benchmark.modal.run/benchmark
```

## 4) Auto-detect endpoint URL

The frontend now auto-fills **Modal endpoint URL** using your active Modal profile in `~/.modal.toml`.
For your current login, this resolves to:

```text
https://manningwu07--deploybench-benchmark.modal.run
```

You can override auto-detection with:

```bash
export DEPLOYBENCH_MODAL_ENDPOINT_URL="https://your-custom-endpoint.modal.run"
```

## 5) If you still see Python 3.9 errors

That means Streamlit was launched outside your project `.venv`.

Use:

```bash
source .venv/bin/activate
which python
python --version
which streamlit
streamlit run frontend/streamlit_app.py
```

`which python` and `which streamlit` should both point to `.venv/bin/...`.
