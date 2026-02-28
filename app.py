"""DeployBench Modal app with job-based API, queued benchmarking, and shared inference gateway."""

import json
import time
from pathlib import Path
from typing import Optional

import modal

app = modal.App("deploybench")

_SOURCE_MODULES = ("workers", "utils", "backend", "orchestrator")

worker_image = (
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
        "python-multipart",
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

api_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi[standard]", "python-multipart")
    .add_local_python_source("backend")
)

artifacts_volume = modal.Volume.from_name("deploybench-artifacts", create_if_missing=True)
data_volume = modal.Volume.from_name("deploybench-data", create_if_missing=True)

LANE_CONFIGS = ["FP32", "FP16", "INT8", "ONNX_FP16"]


@app.function(
    image=worker_image,
    gpu="T4",
    timeout=900,
    volumes={"/data": data_volume, "/artifacts": artifacts_volume},
)
def run_gallery_lane(config: str, model_name: str, task: str, requests_per_day: int) -> dict:
    from workers.pipeline import LaneContext, LaneUnsupportedError, normalize_lane_error, run_lane

    ctx = LaneContext(
        config=config,
        mode="gallery",
        model_ref=f"{model_name}.pt",
        task=task,
        requests_per_day=int(requests_per_day),
    )

    try:
        return run_lane(ctx)
    except LaneUnsupportedError as exc:
        return normalize_lane_error(config, exc, status="unsupported")
    except Exception as exc:  # noqa: BLE001
        return normalize_lane_error(config, exc, status="failed")


@app.function(
    image=worker_image,
    gpu="T4",
    timeout=900,
    volumes={"/artifacts": artifacts_volume, "/data": data_volume},
)
def run_preview_inference(job_id: str, image_bytes: bytes, config: str | None = None) -> dict:
    from workers.preview import run_preview

    return run_preview(job_id=job_id, image_bytes=image_bytes, config=config)


@app.function(
    image=worker_image,
    gpu="T4",
    timeout=900,
    volumes={"/artifacts": artifacts_volume, "/data": data_volume},
)
def run_deployed_inference(deployment_id: str, image_bytes: bytes) -> dict:
    from workers.preview import run_deployment_infer

    return run_deployment_infer(deployment_id=deployment_id, image_bytes=image_bytes)


@app.function(
    image=worker_image,
    timeout=1800,
    concurrency_limit=2,
    volumes={"/artifacts": artifacts_volume, "/data": data_volume},
)
def process_job(job_id: str) -> None:
    from backend.jobs import finalize_job, merge_lane
    from backend.storage import read_job, write_job
    from workers.pipeline import normalize_lane_error

    try:
        artifacts_volume.reload()
    except Exception:  # noqa: BLE001
        pass

    record = read_job(job_id)
    record["status"] = "running"
    for cfg in LANE_CONFIGS:
        lane = record["lanes"][cfg]
        if lane.get("status") == "queued":
            lane["status"] = "running"
    write_job(record)
    artifacts_volume.commit()

    if record["mode"] == "gallery":
        _process_gallery_job(job_id)
    else:
        _process_upload_job(job_id)

    try:
        artifacts_volume.reload()
    except Exception:  # noqa: BLE001
        pass

    final_record = read_job(job_id)
    final_record = finalize_job(final_record)
    write_job(final_record)
    artifacts_volume.commit()



def _process_gallery_job(job_id: str) -> None:
    from backend.jobs import finalize_job, merge_lane
    from backend.storage import read_job, write_job
    from workers.pipeline import normalize_lane_error

    record = read_job(job_id)
    model_name = record.get("model_name") or "yolov8s"
    task = record.get("task") or "detect"
    requests_per_day = int(record.get("requests_per_day", 10000))

    calls = {
        cfg: run_gallery_lane.spawn(cfg, model_name, task, requests_per_day)
        for cfg in LANE_CONFIGS
    }
    pending = set(LANE_CONFIGS)

    while pending:
        for cfg in list(pending):
            call = calls[cfg]
            try:
                lane_payload = call.get(timeout=0)
            except TimeoutError:
                continue
            except Exception as exc:  # noqa: BLE001
                lane_payload = normalize_lane_error(cfg, exc, status="failed")

            record = read_job(job_id)
            record = merge_lane(record, lane_payload)
            record = finalize_job(record)
            write_job(record)
            artifacts_volume.commit()
            pending.remove(cfg)

        if pending:
            time.sleep(1.0)



def _process_upload_job(job_id: str) -> None:
    from backend.jobs import finalize_job, merge_lane
    from backend.storage import read_job, write_job
    from workers.pipeline import normalize_lane_error

    record = read_job(job_id)
    model_ref = record["model_ref"]
    task = record.get("task") or "detect"
    requests_per_day = int(record.get("requests_per_day", 10000))

    lane_out_dir = Path("/artifacts") / "jobs" / job_id / "lanes"
    lane_out_dir.mkdir(parents=True, exist_ok=True)

    sandboxes: dict[str, modal.Sandbox] = {}
    lane_files: dict[str, Path] = {}

    for cfg in LANE_CONFIGS:
        out_path = lane_out_dir / f"{cfg}.json"
        lane_files[cfg] = out_path
        sb = modal.Sandbox.create(
            "python",
            "-m",
            "workers.sandbox_entrypoint",
            "--job-id",
            job_id,
            "--config",
            cfg,
            "--model-ref",
            model_ref,
            "--task",
            task,
            "--requests-per-day",
            str(requests_per_day),
            "--out",
            str(out_path),
            app=app,
            image=worker_image,
            gpu="T4",
            timeout=900,
            volumes={"/artifacts": artifacts_volume, "/data": data_volume},
        )
        sandboxes[cfg] = sb

    pending = set(LANE_CONFIGS)

    while pending:
        for cfg in list(pending):
            sb = sandboxes[cfg]
            rc = sb.returncode
            if rc is None:
                continue

            lane_payload = None
            out_path = lane_files[cfg]

            if out_path.exists():
                try:
                    lane_payload = json.loads(out_path.read_text(encoding="utf-8"))
                except Exception as exc:  # noqa: BLE001
                    lane_payload = normalize_lane_error(
                        cfg,
                        RuntimeError(f"Could not parse sandbox output: {exc}"),
                        status="failed",
                    )
            else:
                stderr = ""
                try:
                    stderr = sb.stderr.read() or ""
                except Exception:  # noqa: BLE001
                    stderr = ""
                msg = f"Sandbox exited with code {rc}."
                if stderr:
                    msg += f" stderr={stderr.strip()[:300]}"
                lane_payload = normalize_lane_error(cfg, RuntimeError(msg), status="failed")

            record = read_job(job_id)
            record = merge_lane(record, lane_payload)
            record = finalize_job(record)
            write_job(record)
            artifacts_volume.commit()
            pending.remove(cfg)

        if pending:
            time.sleep(1.0)


@app.function(image=api_image, timeout=1200, volumes={"/artifacts": artifacts_volume})
@modal.asgi_app()
def api_app():
    from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
    from pydantic import BaseModel

    from backend.adapters import detect_adapter
    from backend.auth import generate_api_key, hash_api_key, verify_api_key
    from backend.jobs import create_job_record, finalize_job, public_job
    from backend.storage import (
        ensure_dirs,
        new_id,
        persist_sample_image,
        persist_upload,
        read_deployment,
        read_job,
        utc_now_iso,
        write_deployment,
        write_job,
    )

    web = FastAPI(title="DeployBench API", version="1.0.0")

    class JobAccepted(BaseModel):
        job_id: str
        status_url: str

    class DeployRequest(BaseModel):
        config: str | None = None

    class DeploymentResponse(BaseModel):
        deployment_id: str
        url: str
        api_key_once: str
        curl_example: str
        response_schema: dict

    @web.post("/v1/jobs", response_model=JobAccepted, status_code=202)
    async def create_job(
        mode: str = Form(...),
        requests_per_day: int = Form(10000),
        task_hint: str | None = Form(None),
        model_name: str | None = Form(None),
        model_file: Optional[UploadFile] = File(None),
        sample_image: Optional[UploadFile] = File(None),
    ):
        if mode not in {"gallery", "upload"}:
            raise HTTPException(status_code=400, detail="mode must be 'gallery' or 'upload'")
        if requests_per_day <= 0:
            raise HTTPException(status_code=400, detail="requests_per_day must be > 0")

        ensure_dirs()
        try:
            artifacts_volume.reload()
        except Exception:  # noqa: BLE001
            pass

        job_id = new_id("job")
        sample_image_ref = None

        if sample_image is not None:
            sample_blob = await sample_image.read()
            sample_image_ref = persist_sample_image(job_id, sample_image.filename or "sample.jpg", sample_blob)

        resolved_task = task_hint
        model_ref = None

        if mode == "gallery":
            if not model_name:
                raise HTTPException(status_code=400, detail="model_name is required in gallery mode")
            model_ref = f"{model_name}.pt"
            if not resolved_task:
                resolved_task = "detect"
        else:
            if model_file is None:
                raise HTTPException(status_code=400, detail="model_file is required in upload mode")
            model_blob = await model_file.read()
            if len(model_blob) > 500 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="model_file exceeds 500MB limit")
            model_ref = persist_upload(job_id, model_file.filename or "model.pt", model_blob)

            adapter_check = detect_adapter(model_ref, task_hint=task_hint)
            if not adapter_check.compatible:
                record = create_job_record(
                    job_id=job_id,
                    mode=mode,
                    requests_per_day=requests_per_day,
                    task_hint=task_hint,
                    model_name=None,
                    model_ref=model_ref,
                    sample_image_ref=sample_image_ref,
                )
                record["status"] = "incompatible"
                record["errors"].append(
                    {
                        "code": "INCOMPATIBLE_MODEL",
                        "message": adapter_check.reason,
                        "adapter": adapter_check.adapter,
                        "task": adapter_check.task,
                    }
                )
                for cfg in LANE_CONFIGS:
                    record["lanes"][cfg]["status"] = "unsupported"
                    record["lanes"][cfg]["error"] = adapter_check.reason
                write_job(record)
                artifacts_volume.commit()
                return JobAccepted(job_id=job_id, status_url=f"/v1/jobs/{job_id}")

            resolved_task = adapter_check.task

        record = create_job_record(
            job_id=job_id,
            mode=mode,
            requests_per_day=requests_per_day,
            task_hint=resolved_task,
            model_name=model_name,
            model_ref=model_ref,
            sample_image_ref=sample_image_ref,
        )
        write_job(record)
        artifacts_volume.commit()

        process_job.spawn(job_id)
        return JobAccepted(job_id=job_id, status_url=f"/v1/jobs/{job_id}")

    @web.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str):
        try:
            artifacts_volume.reload()
        except Exception:  # noqa: BLE001
            pass

        try:
            job = read_job(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return public_job(job)

    @web.post("/v1/jobs/{job_id}/preview")
    async def preview_job(
        job_id: str,
        image: UploadFile = File(...),
        config: str | None = Form(None),
    ):
        try:
            artifacts_volume.reload()
        except Exception:  # noqa: BLE001
            pass

        try:
            _ = read_job(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        payload = run_preview_inference.remote(job_id, await image.read(), config)
        return payload

    @web.post("/v1/jobs/{job_id}/deploy", response_model=DeploymentResponse)
    async def deploy_job(job_id: str, req: DeployRequest, x_forwarded_host: str | None = Header(default=None)):
        try:
            artifacts_volume.reload()
        except Exception:  # noqa: BLE001
            pass

        try:
            record = read_job(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        record = finalize_job(record)
        requested_config = req.config
        if requested_config is None:
            rec = record.get("recommendation")
            if not rec:
                raise HTTPException(status_code=400, detail="No recommendation available for this job")
            requested_config = rec.get("best_config")

        lane = record["lanes"].get(requested_config)
        if not lane or lane.get("status") != "done":
            raise HTTPException(
                status_code=400,
                detail=f"Config '{requested_config}' is not deployable (status must be done)",
            )

        deployment_id = new_id("dep")
        api_key = generate_api_key()

        deployment = {
            "deployment_id": deployment_id,
            "job_id": job_id,
            "config": requested_config,
            "task": record.get("task") or "detect",
            "artifact_paths": {
                "model_ref": record.get("model_ref"),
                "sample_image_ref": record.get("sample_image_ref"),
            },
            "api_key_hash": hash_api_key(api_key),
            "created_at": utc_now_iso(),
            "status": "active",
        }
        write_deployment(deployment)

        record["deployment"] = {
            "deployment_id": deployment_id,
            "config": requested_config,
            "created_at": deployment["created_at"],
            "status": "active",
        }
        write_job(record)
        artifacts_volume.commit()

        base = f"https://{x_forwarded_host}" if x_forwarded_host else "https://your-modal-endpoint.modal.run"
        infer_url = f"{base}/v1/infer/{deployment_id}"

        curl = (
            "curl -X POST "
            + infer_url
            + " -H 'x-api-key: "
            + api_key
            + "' -F 'image=@sample.jpg'"
        )

        return DeploymentResponse(
            deployment_id=deployment_id,
            url=infer_url,
            api_key_once=api_key,
            curl_example=curl,
            response_schema={
                "deployment_id": "string",
                "config": "FP32|FP16|INT8|ONNX_FP16",
                "task": "detect|classify|segment",
                "outputs": "task-specific inference payload",
            },
        )

    @web.post("/v1/infer/{deployment_id}")
    async def infer(deployment_id: str, image: UploadFile = File(...), x_api_key: str | None = Header(default=None)):
        try:
            artifacts_volume.reload()
        except Exception:  # noqa: BLE001
            pass

        try:
            deployment = read_deployment(deployment_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        if not x_api_key or not verify_api_key(x_api_key, deployment.get("api_key_hash", "")):
            raise HTTPException(status_code=401, detail="Invalid API key")

        result = run_deployed_inference.remote(deployment_id, await image.read())
        return result

    @web.get("/healthz")
    async def healthcheck():
        return {"ok": True}

    return web
