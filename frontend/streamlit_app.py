"""
DeployBench — Streamlit frontend.

Run locally:   streamlit run frontend/streamlit_app.py
Deploy on Modal: see app.py for the @modal.asgi_app() wrapper.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
import streamlit as st
import requests
import plotly.graph_objects as go
from urllib.parse import urlparse


def _detect_modal_workspace() -> str | None:
    """Infer Modal workspace/login from env or local Modal config."""
    env_workspace = (
        os.getenv("MODAL_WORKSPACE")
        or os.getenv("MODAL_PROFILE")
        or os.getenv("MODAL_USERNAME")
    )
    if env_workspace:
        return env_workspace.strip()

    config_path = Path.home() / ".modal.toml"
    if not config_path.exists():
        return None

    text = config_path.read_text(encoding="utf-8")
    active_profile_match = re.search(
        r"(?ms)^\[(?P<name>[^\]]+)\]\s+.*?^active\s*=\s*true\b",
        text,
    )
    if active_profile_match:
        return active_profile_match.group("name").strip()

    first_profile_match = re.search(r"(?m)^\[([^\]]+)\]", text)
    if first_profile_match:
        return first_profile_match.group(1).strip()
    return None


def _default_endpoint_url() -> str:
    """Return best default endpoint, preferring explicit env override."""
    explicit_url = os.getenv("DEPLOYBENCH_MODAL_ENDPOINT_URL")
    if explicit_url:
        return explicit_url.strip()

    workspace = _detect_modal_workspace()
    if workspace:
        return f"https://{workspace}--deploybench-benchmark.modal.run"
    return "https://your-modal-endpoint.modal.run"

# ---------------------------------------------------------------------------
# Page config (must be the very first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="DeployBench", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar — configuration and dev utilities
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    mock_mode = st.toggle("Mock data mode", value=False,
                          help="Use fake results so you can develop the UI without a GPU.")
    endpoint_url = st.text_input(
        "Modal endpoint URL",
        value=_default_endpoint_url(),
        disabled=mock_mode,
    )
    st.divider()
    st.caption("DeployBench — HackIllinois 2026")

# ---------------------------------------------------------------------------
# Mock data (used when mock_mode is True)
# ---------------------------------------------------------------------------
_MOCK_RESPONSE = {
    "model": "yolov8s",
    "results": [
        {
            "config": "FP32 (Baseline)",
            "avg_latency_ms": 42.1,
            "p95_latency_ms": 55.3,
            "fps": 23.8,
            "peak_memory_mb": 1240.0,
            "mAP_50": 0.483,
            "est_monthly_cost": 88.24,
        },
        {
            "config": "FP16 (Half Precision)",
            "avg_latency_ms": 28.5,
            "p95_latency_ms": 35.1,
            "fps": 35.1,
            "peak_memory_mb": 680.0,
            "mAP_50": 0.479,
            "est_monthly_cost": 59.75,
        },
        {
            "config": "INT8 (Quantized)",
            "error": "quantize_dynamic not supported on YOLO conv layers",
        },
        {
            "config": "ONNX + FP16",
            "avg_latency_ms": 19.2,
            "p95_latency_ms": 24.0,
            "fps": 52.1,
            "peak_memory_mb": 510.0,
            "mAP_50": 0.471,
            "est_monthly_cost": 40.27,
        },
    ],
    "recommendation": {
        "best_config": "ONNX + FP16",
        "monthly_savings": 47.97,
        "accuracy_tradeoff": 0.012,
        "speedup": 2.19,
    },
    "deploy_script": (
        'import modal\nfrom ultralytics import YOLO\n\n'
        'app = modal.App("my-yolov8s-deployment")\n# ... (generated script)'
    ),
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("DeployBench")
st.caption("Find the cheapest way to deploy your vision model in 60 seconds.")

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
col1, col2 = st.columns([2, 1])
with col1:
    model = st.selectbox("Select Model", ["yolov8n", "yolov8s", "yolov8m"])
with col2:
    requests_per_day = st.number_input("Daily requests", value=10_000, step=1_000, min_value=1)

run = st.button("Run Benchmark", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_get(d: dict, key: str, default=None):
    """Return d[key] if present, else default."""
    return d.get(key, default)


def _is_error_result(r: dict) -> bool:
    return "error" in r


def _valid_results(results: list[dict]) -> list[dict]:
    return [r for r in results if not _is_error_result(r)]


def _benchmark_url_candidates(raw_url: str) -> list[str]:
    """
    Return one or two endpoint candidates.
    Accepts either a direct web endpoint URL or a base app URL.
    """
    url = raw_url.strip().rstrip("/")
    if not url or "your-modal-endpoint.modal.run" in url:
        raise ValueError("Set your real Modal endpoint URL in the sidebar.")

    parsed = urlparse(url)
    path = parsed.path or ""

    # Modal web-function URLs commonly work at root on a benchmark-specific
    # subdomain, but some setups use "/benchmark".
    if path.endswith("/benchmark"):
        return [url, url[: -len("/benchmark")]]
    if path and path != "/":
        return [url]
    return [url, f"{url}/benchmark"]


def _render_metric_cards(results: list[dict]) -> None:
    cols = st.columns(max(len(results), 1))
    for i, r in enumerate(results):
        config = _safe_get(r, "config", f"Config {i+1}")
        if _is_error_result(r):
            cols[i].error(f"**{config}**\nFailed: {r['error']}")
            continue
        with cols[i]:
            cost = _safe_get(r, "est_monthly_cost")
            cost_label = f"${cost}/mo" if cost is not None else "N/A"
            st.metric(config, cost_label)
            latency = _safe_get(r, "avg_latency_ms", "?")
            fps = _safe_get(r, "fps", "?")
            mAP = _safe_get(r, "mAP_50")
            mem = _safe_get(r, "peak_memory_mb", "?")
            st.caption(f"{latency} ms · {fps} FPS")
            map_str = f"{mAP:.1%}" if mAP is not None else "N/A"
            st.caption(f"mAP: {map_str} · {mem} MB")


def _render_cost_accuracy_chart(valid: list[dict]) -> None:
    if not valid:
        st.info("No valid results to chart.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[_safe_get(r, "est_monthly_cost", 0) for r in valid],
        y=[_safe_get(r, "mAP_50", 0) for r in valid],
        mode="markers+text",
        text=[_safe_get(r, "config", "") for r in valid],
        textposition="top center",
        marker=dict(size=15, color="#4d96ff"),
    ))
    fig.update_layout(
        xaxis_title="Estimated Monthly Cost ($)",
        yaxis_title="Accuracy (mAP@50)",
        height=400,
        margin=dict(t=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_cost_bar_chart(valid: list[dict]) -> None:
    if not valid:
        return
    colors = ["#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff"]
    fig = go.Figure(go.Bar(
        x=[_safe_get(r, "config", f"Config {i}") for i, r in enumerate(valid)],
        y=[_safe_get(r, "est_monthly_cost", 0) for r in valid],
        marker_color=colors[: len(valid)],
    ))
    fig.update_layout(
        yaxis_title="Est. Monthly Cost ($)",
        height=350,
        margin=dict(t=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_recommendation(rec: dict | None) -> None:
    if rec is None:
        st.warning("No recommendation available (all workers may have failed).")
        return
    best = _safe_get(rec, "best_config", "Unknown")
    savings = _safe_get(rec, "monthly_savings")
    speedup = _safe_get(rec, "speedup")
    tradeoff = _safe_get(rec, "accuracy_tradeoff")

    savings_str = f"${savings}/mo" if savings is not None else "N/A"
    speedup_str = f"{speedup}x faster" if speedup is not None else ""
    tradeoff_str = f"{tradeoff:.2%} accuracy tradeoff" if tradeoff is not None else ""

    parts = [p for p in [speedup_str, tradeoff_str] if p]
    detail = ", ".join(parts)
    msg = f"**Recommendation: {best}** — Saves {savings_str}"
    if detail:
        msg += f" ({detail})"
    st.success(msg)


def _render_deploy_script(script: str | None) -> None:
    if not script:
        st.info("No deployment script generated.")
        return
    st.caption("Copy this script, save as `modal_app.py`, then run `modal deploy modal_app.py`")
    st.code(script, language="python")


# ---------------------------------------------------------------------------
# Main results section
# ---------------------------------------------------------------------------
if run:
    if mock_mode:
        data = _MOCK_RESPONSE
    else:
        with st.spinner("Spinning up 4 GPU containers on Modal..."):
            try:
                candidates = _benchmark_url_candidates(endpoint_url)
                last_http_error = None
                for benchmark_url in candidates:
                    resp = requests.post(
                        benchmark_url,
                        json={"model": model, "requests_per_day": int(requests_per_day)},
                        timeout=660,
                    )
                    try:
                        resp.raise_for_status()
                        data = resp.json()
                        break
                    except requests.exceptions.HTTPError as exc:
                        if exc.response is not None and exc.response.status_code == 404:
                            last_http_error = exc
                            continue
                        raise
                else:
                    if last_http_error is not None:
                        raise last_http_error
                    raise requests.exceptions.HTTPError("No endpoint URL candidates succeeded.")
            except ValueError as exc:
                st.error(str(exc))
                st.stop()
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the Modal endpoint. Check the URL in the sidebar.")
                st.stop()
            except requests.exceptions.Timeout:
                st.error("Request timed out. The benchmark may still be running — try again.")
                st.stop()
            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    st.error(
                        "Endpoint returned 404. In the sidebar, use your deployed Modal URL "
                        "(base URL or full /benchmark URL)."
                    )
                    st.stop()
                st.error(f"Endpoint returned an error: {exc}")
                st.stop()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Unexpected error: {exc}")
                st.stop()

    # Validate top-level keys
    if "results" not in data:
        st.error("Invalid response from server: missing 'results' key.")
        st.stop()

    results: list[dict] = data["results"]

    # --- Metric cards ---
    st.subheader("Results")
    _render_metric_cards(results)

    valid = _valid_results(results)

    # --- Charts (only render if we have at least one successful result) ---
    if valid:
        st.subheader("Cost vs Accuracy Tradeoff")
        _render_cost_accuracy_chart(valid)

        st.subheader("Monthly Cost Comparison")
        _render_cost_bar_chart(valid)
    else:
        st.warning("All workers failed — no chart data to display.")

    # --- Recommendation ---
    st.subheader("Recommendation")
    _render_recommendation(_safe_get(data, "recommendation"))

    # --- Deployment script ---
    st.subheader("Deploy to Modal")
    _render_deploy_script(_safe_get(data, "deploy_script"))
