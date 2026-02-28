"""Shared pytest fixtures and utilities for DeployBench tests."""
import numpy  # noqa: F401
import pytest


class MockCall:
    """
    Simulates a Modal spawned function call's .get() interface.

    Usage:
        # Successful call
        call = MockCall(return_value={"config": "FP32", ...})
        result = call.get()  # -> {"config": "FP32", ...}

        # Failing call
        call = MockCall(raises=RuntimeError("GPU OOM"))
        call.get()  # -> raises RuntimeError
    """

    def __init__(self, return_value=None, raises=None):
        self._return_value = return_value
        self._raises = raises

    def get(self):
        if self._raises is not None:
            raise self._raises
        return self._return_value


def make_worker_result(
    config: str,
    avg_latency_ms: float = 100.0,
    p95_latency_ms: float = 120.0,
    fps: float = 10.0,
    peak_memory_mb: float = 1000.0,
    mAP_50: float = 0.45,
) -> dict:
    """Build a mock worker result dict matching the spec schema."""
    return {
        "config": config,
        "avg_latency_ms": avg_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "fps": fps,
        "peak_memory_mb": peak_memory_mb,
        "mAP_50": mAP_50,
    }


@pytest.fixture
def fp32_result():
    return make_worker_result("FP32 (Baseline)", avg_latency_ms=100.0, mAP_50=0.50)


@pytest.fixture
def fp16_result():
    return make_worker_result("FP16 (Half Precision)", avg_latency_ms=60.0, mAP_50=0.49)


@pytest.fixture
def int8_result():
    return make_worker_result("INT8 (Quantized)", avg_latency_ms=40.0, mAP_50=0.46)


@pytest.fixture
def onnx_result():
    return make_worker_result("ONNX + FP16", avg_latency_ms=45.0, mAP_50=0.48)


@pytest.fixture
def all_worker_results(fp32_result, fp16_result, int8_result, onnx_result):
    return [fp32_result, fp16_result, int8_result, onnx_result]


@pytest.fixture
def sample_coco_ground_truth():
    """Minimal valid COCO-format ground truth dict for mAP tests."""
    return {
        "images": [
            {"id": 1, "file_name": "000000001.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "000000002.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100.0, 100.0, 50.0, 50.0],
                "area": 2500.0,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [200.0, 200.0, 80.0, 80.0],
                "area": 6400.0,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 1, "name": "person", "supercategory": "person"},
        ],
    }


@pytest.fixture
def sample_coco_predictions():
    """Sample COCO-format predictions (perfect match for ground truth above)."""
    return [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [100.0, 100.0, 50.0, 50.0],
            "score": 0.95,
        },
        {
            "image_id": 2,
            "category_id": 1,
            "bbox": [200.0, 200.0, 80.0, 80.0],
            "score": 0.90,
        },
    ]
