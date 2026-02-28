"""DeployBench Streamlit frontend for job-based API workflow."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import streamlit as st


# ---------------------------------------------------------------------------
# Endpoint helpers
# ---------------------------------------------------------------------------

def _detect_modal_workspace() -> str | None:
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
    explicit_url = os.getenv("DEPLOYBENCH_MODAL_ENDPOINT_URL")
    if explicit_url:
        return explicit_url.strip().rstrip("/")

    workspace = _detect_modal_workspace()
    if workspace:
        return f"https://{workspace}--deploybench-api-app.modal.run"
    return "https://your-modal-endpoint.modal.run"



def _normalize_base_url(raw: str) -> str:
    value = (raw or "").strip().rstrip("/")
    if not value:
        raise ValueError("Set Modal endpoint URL in sidebar")

    parsed = urlparse(value)
    if not parsed.scheme:
        value = f"https://{value}"
    return value.rstrip("/")



def _api_url(base: str, path: str) -> str:
    return f"{base}{path}"


def _trigger_rerun() -> None:
    """Streamlit rerun compatibility across versions."""
    rerun_fn = getattr(st, "rerun", None)
    if callable(rerun_fn):
        rerun_fn()
        return

    legacy_fn = getattr(st, "experimental_rerun", None)
    if callable(legacy_fn):
        legacy_fn()
        return

    raise RuntimeError("Your Streamlit version does not support rerun APIs.")


def _format_http_error(exc: Exception) -> str:
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        body = exc.response.text.strip()
        if len(body) > 400:
            body = body[:400] + "..."
        return f"{exc} | body: {body}"
    return str(exc)



def _post_job(base_url: str, *, mode: str, requests_per_day: int,
              model_name: str | None = None, task_hint: str | None = None,
              model_file=None, sample_image=None) -> dict[str, Any]:
    data = {
        "mode": mode,
        "requests_per_day": str(int(requests_per_day)),
    }
    if task_hint:
        data["task_hint"] = task_hint
    if model_name:
        data["model_name"] = model_name

    files = {}
    if model_file is not None:
        files["model_file"] = (model_file.name, model_file.getvalue(), "application/octet-stream")
    if sample_image is not None:
        files["sample_image"] = (sample_image.name, sample_image.getvalue(), sample_image.type or "image/jpeg")

    resp = requests.post(
        _api_url(base_url, "/v1/jobs"),
        data=data,
        files=files or None,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()



def _get_job(base_url: str, job_id: str) -> dict[str, Any]:
    resp = requests.get(_api_url(base_url, f"/v1/jobs/{job_id}"), timeout=30)
    resp.raise_for_status()
    return resp.json()



def _preview_job(base_url: str, job_id: str, image_file, config: str | None) -> dict[str, Any]:
    data = {}
    if config:
        data["config"] = config

    files = {
        "image": (image_file.name, image_file.getvalue(), image_file.type or "image/jpeg"),
    }

    resp = requests.post(
        _api_url(base_url, f"/v1/jobs/{job_id}/preview"),
        data=data,
        files=files,
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()



def _deploy_job(base_url: str, job_id: str, config: str | None = None) -> dict[str, Any]:
    payload = {"config": config} if config else {}
    resp = requests.post(
        _api_url(base_url, f"/v1/jobs/{job_id}/deploy"),
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()



def _invoke_deployment(url: str, api_key: str, image_file) -> dict[str, Any]:
    files = {"image": (image_file.name, image_file.getvalue(), image_file.type or "image/jpeg")}
    headers = {"x-api-key": api_key}
    resp = requests.post(url, headers=headers, files=files, timeout=120)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="DeployBench", layout="wide")

with st.sidebar:
    st.header("Settings")
    endpoint_url = st.text_input("Modal endpoint URL", value=_default_endpoint_url())
    st.caption("Use the base API URL. Example: https://<workspace>--deploybench-api-app.modal.run")
    st.divider()
    st.caption("Cost assumptions: T4 GPU, 30-day month, 1.15 overhead factor")

st.title("DeployBench")
st.caption("Upload your model, benchmark optimizations in parallel, and deploy a live API.")

if "active_job_id" not in st.session_state:
    st.session_state.active_job_id = None
if "last_job" not in st.session_state:
    st.session_state.last_job = None
if "deployment" not in st.session_state:
    st.session_state.deployment = None

base_url = None
try:
    base_url = _normalize_base_url(endpoint_url)
except ValueError as exc:
    st.warning(str(exc))

requests_per_day = st.number_input("Expected requests/day", min_value=1, value=10000, step=1000)

try_tab, upload_tab = st.tabs(["Try It", "Upload Your Model"])

with try_tab:
    st.subheader("Gallery path")
    gallery_model = st.selectbox("Model", ["yolov8n", "yolov8s", "yolov8m"], key="gallery_model")
    gallery_sample = st.file_uploader("Optional sample image", type=["jpg", "jpeg", "png"], key="gallery_sample")
    if st.button("Run Gallery Benchmark", type="primary", use_container_width=True):
        if not base_url:
            st.error("Set endpoint URL first.")
        else:
            created = None
            try:
                created = _post_job(
                    base_url,
                    mode="gallery",
                    requests_per_day=int(requests_per_day),
                    model_name=gallery_model,
                    sample_image=gallery_sample,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to create job: {_format_http_error(exc)}")

            if created:
                st.session_state.active_job_id = created["job_id"]
                st.session_state.last_job = None
                st.session_state.deployment = None
                st.success(f"Job created: {created['job_id']}")
                try:
                    _trigger_rerun()
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"Auto-refresh unavailable: {exc}. Refresh the page manually.")

with upload_tab:
    st.subheader("Upload path")
    upload_model = st.file_uploader("Upload .pt model (max 500MB)", type=["pt"], key="upload_model")
    task_hint = st.selectbox("Task hint (optional)", ["", "detect", "classify", "segment"], index=0)
    upload_sample = st.file_uploader("Optional sample image", type=["jpg", "jpeg", "png"], key="upload_sample")

    if st.button("Run Upload Benchmark", type="primary", use_container_width=True):
        if not base_url:
            st.error("Set endpoint URL first.")
        elif upload_model is None:
            st.error("Upload a .pt model first.")
        else:
            created = None
            try:
                created = _post_job(
                    base_url,
                    mode="upload",
                    requests_per_day=int(requests_per_day),
                    task_hint=task_hint or None,
                    model_file=upload_model,
                    sample_image=upload_sample,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to create job: {_format_http_error(exc)}")

            if created:
                st.session_state.active_job_id = created["job_id"]
                st.session_state.last_job = None
                st.session_state.deployment = None
                st.success(f"Job created: {created['job_id']}")
                try:
                    _trigger_rerun()
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"Auto-refresh unavailable: {exc}. Refresh the page manually.")

# ---------------------------------------------------------------------------
# Active job panel + polling
# ---------------------------------------------------------------------------
job_id = st.session_state.active_job_id
if job_id and base_url:
    st.divider()
    st.subheader(f"Live Job: {job_id}")

    try:
        job = _get_job(base_url, job_id)
        st.session_state.last_job = job
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not fetch job status: {exc}")
        st.stop()

    status = job.get("status", "unknown")
    st.write(f"Status: `{status}`")

    lanes = job.get("lanes", {})
    lane_cols = st.columns(4)
    for idx, cfg in enumerate(["FP32", "FP16", "INT8", "ONNX_FP16"]):
        lane = lanes.get(cfg, {})
        with lane_cols[idx]:
            st.markdown(f"**{cfg}**")
            st.caption(f"state: {lane.get('status', 'unknown')}")
            if lane.get("status") == "done":
                st.metric("Latency", f"{lane.get('avg_latency_ms', 'n/a')} ms")
                st.caption(f"p95: {lane.get('p95_latency_ms', 'n/a')} ms")
                st.caption(f"FPS: {lane.get('fps', 'n/a')}")
                st.caption(f"Memory: {lane.get('peak_memory_mb', 'n/a')} MB")
                st.caption(f"Cost: ${lane.get('est_monthly_cost', 'n/a')}/mo")
                gpu_avg = lane.get("gpu_util_avg_pct")
                gpu_p95 = lane.get("gpu_util_p95_pct")
                gpu_peak = lane.get("gpu_util_peak_pct")
                samples = lane.get("gpu_monitor_sample_count", 0)
                if gpu_avg is not None:
                    st.caption(
                        f"GPU util avg/p95/peak: {gpu_avg}% / {gpu_p95}% / {gpu_peak}%"
                    )
                    st.caption(
                        f"GPU mem peak: {lane.get('gpu_mem_used_peak_mb', 'n/a')} MB "
                        f"({lane.get('gpu_mem_used_peak_pct', 'n/a')}%)"
                    )
                    st.caption(
                        f"GPU power avg: {lane.get('gpu_power_avg_w', 'n/a')} W | "
                        f"samples: {samples}"
                    )
                elif samples == 0 and lane.get("gpu_monitor_errors"):
                    st.caption(
                        "GPU telemetry unavailable: "
                        + "; ".join(lane.get("gpu_monitor_errors", [])[:2])
                    )
                q_name = lane.get("quality_metric_name") or "quality"
                q_val = lane.get("quality_metric")
                if q_val is not None:
                    st.caption(f"{q_name}: {q_val}")
            elif lane.get("status") in {"failed", "unsupported"}:
                st.error(lane.get("error") or "Lane failed")

    rec = job.get("recommendation")
    if rec:
        st.success(
            f"Recommendation: {rec.get('best_config')} | "
            f"Savings: ${rec.get('monthly_savings')} /mo | "
            f"Speedup: {rec.get('speedup_vs_fp32')}x | "
            f"Quality tradeoff: {rec.get('quality_tradeoff')} | "
            f"Confidence: {rec.get('confidence')}"
        )
        st.caption(rec.get("decision_reason", ""))

    if job.get("errors"):
        st.warning("Job errors")
        st.json(job["errors"])

    terminal = status in {"completed", "failed", "incompatible"}
    if not terminal:
        st.info("Live race polling every ~1.5 seconds...")
        time.sleep(1.5)
        try:
            _trigger_rerun()
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Auto-refresh unavailable: {exc}. Refresh the page manually.")

# ---------------------------------------------------------------------------
# Completed job actions
# ---------------------------------------------------------------------------
job = st.session_state.last_job
if job and job.get("status") in {"completed", "incompatible", "failed"} and base_url:
    st.divider()
    st.subheader("Post-Benchmark Actions")

    if job.get("status") == "completed":
        done_configs = [
            cfg
            for cfg, lane in (job.get("lanes") or {}).items()
            if lane.get("status") == "done"
        ]

        with st.expander("Preview outputs", expanded=True):
            preview_image = st.file_uploader("Upload image for preview", type=["jpg", "jpeg", "png"], key="preview_image")
            preview_config = st.selectbox(
                "Preview config",
                ["(all done configs)"] + done_configs,
                index=0,
                key="preview_config",
            )
            if st.button("Run Preview", use_container_width=True):
                if preview_image is None:
                    st.error("Upload an image first.")
                else:
                    try:
                        cfg = None if preview_config == "(all done configs)" else preview_config
                        preview = _preview_job(base_url, job["job_id"], preview_image, cfg)
                        for cfg_name, payload in preview.get("results", {}).items():
                            st.markdown(f"**{cfg_name}**")
                            if payload.get("status") in {"failed", "unsupported"}:
                                st.error(payload.get("error", "Preview failed"))
                                continue
                            b64 = payload.get("annotated_image_b64")
                            if b64:
                                st.image(f"data:image/png;base64,{b64}", caption=f"{cfg_name} annotated")
                            st.json(payload.get("outputs", {}))
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Preview failed: {exc}")

        with st.expander("Deploy as API", expanded=True):
            rec = job.get("recommendation") or {}
            default_cfg = rec.get("best_config")
            deploy_config = st.selectbox(
                "Deploy config",
                ["(recommended)"] + done_configs,
                index=0,
                key="deploy_config",
            )
            if st.button("Deploy now", type="primary", use_container_width=True):
                try:
                    cfg = None if deploy_config == "(recommended)" else deploy_config
                    dep = _deploy_job(base_url, job["job_id"], cfg)
                    st.session_state.deployment = dep
                    st.success("Deployment created.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Deploy failed: {exc}")

            dep = st.session_state.deployment
            if dep:
                st.code(dep.get("url", ""), language="text")
                st.code(dep.get("curl_example", ""), language="bash")
                st.warning("API key (shown once):")
                st.code(dep.get("api_key_once", ""), language="text")

                probe_image = st.file_uploader("Smoke test deployed API", type=["jpg", "jpeg", "png"], key="probe_image")
                if st.button("Run deployment smoke test", use_container_width=True):
                    if probe_image is None:
                        st.error("Upload test image first.")
                    else:
                        try:
                            output = _invoke_deployment(dep["url"], dep["api_key_once"], probe_image)
                            st.success("Deployment responded successfully.")
                            st.json(output)
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Smoke test failed: {exc}")

    else:
        st.warning("Job did not complete successfully. Review lane errors above.")
