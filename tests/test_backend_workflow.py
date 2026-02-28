"""Unit tests for new backend workflow components."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.auth import generate_api_key, hash_api_key, verify_api_key
from backend.costs import estimate_monthly_cost
from backend.jobs import create_job_record, finalize_job, merge_lane
from backend.recommendation import build_recommendation
from backend.adapters import detect_adapter


class TestAuth:
    def test_generate_and_verify_api_key(self):
        key = generate_api_key()
        h = hash_api_key(key)
        assert verify_api_key(key, h)
        assert not verify_api_key("wrong", h)

    def test_hash_rejects_empty(self):
        with pytest.raises(ValueError):
            hash_api_key("")


class TestCosts:
    def test_estimate_monthly_cost_includes_overhead(self):
        base = estimate_monthly_cost(100.0, 10_000, overhead_factor=1.0)
        overhead = estimate_monthly_cost(100.0, 10_000, overhead_factor=1.15)
        assert overhead > base


class TestRecommendation:
    def test_quality_safety_gate_prefers_safe_lane(self):
        lanes = {
            "FP32": {
                "config": "FP32",
                "status": "done",
                "est_monthly_cost": 100.0,
                "avg_latency_ms": 100.0,
                "quality_metric": 1.0,
            },
            "FP16": {
                "config": "FP16",
                "status": "done",
                "est_monthly_cost": 60.0,
                "avg_latency_ms": 60.0,
                "quality_metric": 0.995,
            },
            "INT8": {
                "config": "INT8",
                "status": "done",
                "est_monthly_cost": 40.0,
                "avg_latency_ms": 50.0,
                "quality_metric": 0.95,
            },
        }
        rec = build_recommendation(lanes, task="detect")
        assert rec["best_config"] == "FP16"


class TestJobs:
    def test_finalize_job_sets_completed_and_recommendation(self):
        record = create_job_record(
            job_id="job_x",
            mode="upload",
            requests_per_day=10000,
            task_hint="classify",
            model_name=None,
            model_ref="/tmp/model.pt",
            sample_image_ref=None,
        )

        fp32 = {
            "config": "FP32",
            "status": "done",
            "avg_latency_ms": 100.0,
            "est_monthly_cost": 100.0,
            "quality_metric": 1.0,
            "_quality_signature": {"images": {"1": {"top1": 1}}},
        }
        fp16 = {
            "config": "FP16",
            "status": "done",
            "avg_latency_ms": 50.0,
            "est_monthly_cost": 50.0,
            "_quality_signature": {"images": {"1": {"top1": 1}}},
        }
        int8 = {
            "config": "INT8",
            "status": "unsupported",
            "error": "not supported",
        }
        onnx = {
            "config": "ONNX_FP16",
            "status": "failed",
            "error": "boom",
        }

        for lane in [fp32, fp16, int8, onnx]:
            record = merge_lane(record, lane)

        out = finalize_job(record)
        assert out["status"] == "completed"
        assert out["recommendation"] is not None
        assert out["recommendation"]["best_config"] == "FP16"


class TestAdapters:
    def test_rejects_nonexistent_file(self):
        check = detect_adapter("/no/such/model.pt")
        assert not check.compatible
        assert "does not exist" in check.reason

    def test_rejects_non_pt_file(self, tmp_path: Path):
        p = tmp_path / "model.bin"
        p.write_bytes(b"x")
        check = detect_adapter(str(p))
        assert not check.compatible
        assert "Only .pt files" in check.reason

    def test_accepts_supported_ultralytics_task(self, tmp_path: Path):
        p = tmp_path / "model.pt"
        p.write_bytes(b"x")

        fake_ultra = SimpleNamespace(YOLO=lambda _: SimpleNamespace(task="detect"))
        with patch.dict(sys.modules, {"ultralytics": fake_ultra}):
            check = detect_adapter(str(p))

        assert check.compatible
        assert check.adapter == "ultralytics_detect"
