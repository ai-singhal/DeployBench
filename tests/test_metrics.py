"""Tests for utils/metrics.py — compute_mAP() using mocked pycocotools."""
import sys
import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Mock pycocotools before importing utils.metrics
# This prevents GPU/C-extension imports and keeps tests fully local.
# ---------------------------------------------------------------------------

def _make_pycocotools_mock():
    """Build a minimal pycocotools mock that satisfies standard COCO eval usage."""
    mock = MagicMock()

    # COCO(annotation_dict) returns a gt object with loadRes()
    coco_gt_instance = MagicMock()
    coco_dt_instance = MagicMock()
    coco_gt_instance.loadRes.return_value = coco_dt_instance
    mock.coco.COCO.return_value = coco_gt_instance

    # COCOeval(...).stats[1] is mAP@IoU=0.50
    cocoeval_instance = MagicMock()
    cocoeval_instance.stats = [0.40, 0.55, 0.30, 0.10, 0.20, 0.30]  # [mAP, mAP50, ...]
    mock.cocoeval.COCOeval.return_value = cocoeval_instance

    return mock


_pycocotools_mock = _make_pycocotools_mock()
sys.modules.setdefault("pycocotools", _pycocotools_mock)
sys.modules.setdefault("pycocotools.coco", _pycocotools_mock.coco)
sys.modules.setdefault("pycocotools.cocoeval", _pycocotools_mock.cocoeval)

from utils.metrics import compute_mAP  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_pycocotools_mock(mAP50_value: float):
    """Return a pycocotools mock whose COCOeval.stats[1] == mAP50_value."""
    mock = MagicMock()
    gt = MagicMock()
    dt = MagicMock()
    gt.loadRes.return_value = dt
    mock.coco.COCO.return_value = gt

    cocoeval = MagicMock()
    cocoeval.stats = [0.0, mAP50_value, 0.0, 0.0, 0.0, 0.0]
    mock.cocoeval.COCOeval.return_value = cocoeval
    return mock


SAMPLE_GT = {
    "images": [{"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480}],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [100.0, 100.0, 50.0, 50.0],
            "area": 2500.0,
            "iscrowd": 0,
        }
    ],
    "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
}

PERFECT_PREDS = [
    {"image_id": 1, "category_id": 1, "bbox": [100.0, 100.0, 50.0, 50.0], "score": 0.99}
]

WRONG_PREDS = [
    {"image_id": 1, "category_id": 1, "bbox": [0.0, 0.0, 1.0, 1.0], "score": 0.10}
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeMAPReturnType:
    def test_returns_float(self, sample_coco_ground_truth, sample_coco_predictions):
        result = compute_mAP(sample_coco_predictions, sample_coco_ground_truth)
        assert isinstance(result, float)

    def test_result_in_valid_range(self, sample_coco_ground_truth, sample_coco_predictions):
        result = compute_mAP(sample_coco_predictions, sample_coco_ground_truth)
        assert 0.0 <= result <= 1.0


class TestComputeMAPWithMockedCocotools:
    """Patch sys.modules to control the mAP value returned by pycocotools."""

    @pytest.mark.parametrize("expected_map", [0.0, 0.25, 0.50, 0.75, 1.0])
    def test_returns_mAP50_from_cocoeval_stats(self, expected_map):
        """compute_mAP should return stats[1] (mAP@IoU=0.50) from COCOeval."""
        fresh = _fresh_pycocotools_mock(mAP50_value=expected_map)

        with patch.dict(sys.modules, {
            "pycocotools": fresh,
            "pycocotools.coco": fresh.coco,
            "pycocotools.cocoeval": fresh.cocoeval,
        }):
            # Re-import to pick up patched modules (or the function already imported
            # uses the module-level import; patch the attribute directly)
            import importlib
            import utils.metrics as metrics_mod
            importlib.reload(metrics_mod)
            result = metrics_mod.compute_mAP(PERFECT_PREDS, SAMPLE_GT)

        assert result == pytest.approx(expected_map, abs=1e-9)

    def test_cocoeval_evaluate_called(self, sample_coco_ground_truth, sample_coco_predictions):
        """compute_mAP must call evaluate(), accumulate(), and summarize()."""
        cocoeval_instance = _pycocotools_mock.cocoeval.COCOeval.return_value
        compute_mAP(sample_coco_predictions, sample_coco_ground_truth)
        cocoeval_instance.evaluate.assert_called()
        cocoeval_instance.accumulate.assert_called()
        cocoeval_instance.summarize.assert_called()


class TestComputeMAPEdgeCases:
    def test_empty_predictions_returns_zero_or_low(self, sample_coco_ground_truth):
        """With no predictions the mAP must be 0.0 (or pycocotools will return 0)."""
        fresh = _fresh_pycocotools_mock(mAP50_value=0.0)
        with patch.dict(sys.modules, {
            "pycocotools": fresh,
            "pycocotools.coco": fresh.coco,
            "pycocotools.cocoeval": fresh.cocoeval,
        }):
            import importlib
            import utils.metrics as metrics_mod
            importlib.reload(metrics_mod)
            result = metrics_mod.compute_mAP([], sample_coco_ground_truth)
        assert result == 0.0

    def test_accepts_list_of_prediction_dicts(self, sample_coco_ground_truth):
        """predictions must be a list; function should not mutate it."""
        preds = list(PERFECT_PREDS)
        compute_mAP(preds, sample_coco_ground_truth)
        assert preds == PERFECT_PREDS  # unchanged

    def test_accepts_coco_format_ground_truth_dict(self):
        """ground_truth is a plain Python dict, not a file path."""
        result = compute_mAP(PERFECT_PREDS, SAMPLE_GT)
        assert isinstance(result, float)

    def test_ground_truth_with_multiple_categories(self):
        gt = {
            "images": [{"id": 1, "file_name": "img.jpg", "width": 640, "height": 480}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [0, 0, 50, 50], "area": 2500, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 2,
                 "bbox": [100, 100, 30, 30], "area": 900, "iscrowd": 0},
            ],
            "categories": [
                {"id": 1, "name": "person", "supercategory": "person"},
                {"id": 2, "name": "car", "supercategory": "vehicle"},
            ],
        }
        preds = [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 50, 50], "score": 0.9},
            {"image_id": 1, "category_id": 2, "bbox": [100, 100, 30, 30], "score": 0.8},
        ]
        result = compute_mAP(preds, gt)
        assert isinstance(result, float)

    def test_predictions_with_low_confidence(self, sample_coco_ground_truth):
        """Low-confidence predictions are still valid input."""
        low_conf_preds = [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 0.01}
        ]
        result = compute_mAP(low_conf_preds, sample_coco_ground_truth)
        assert isinstance(result, float)

    def test_predictions_for_unknown_image_id(self, sample_coco_ground_truth):
        """Predictions referencing image_ids not in ground truth shouldn't crash."""
        bad_preds = [
            {"image_id": 9999, "category_id": 1, "bbox": [0, 0, 50, 50], "score": 0.9}
        ]
        # Should complete without raising (pycocotools handles this internally)
        result = compute_mAP(bad_preds, sample_coco_ground_truth)
        assert isinstance(result, float)


class TestComputeMAPPredictionFormat:
    """Verify expected prediction dict schema is accepted."""

    def test_prediction_has_required_fields(self, sample_coco_ground_truth):
        """Each prediction dict must have image_id, category_id, bbox, score."""
        preds = [
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": [100.0, 100.0, 50.0, 50.0],
                "score": 0.9,
            }
        ]
        result = compute_mAP(preds, sample_coco_ground_truth)
        assert isinstance(result, float)

    @pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
    def test_various_confidence_scores(self, score, sample_coco_ground_truth):
        preds = [
            {"image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 50], "score": score}
        ]
        result = compute_mAP(preds, sample_coco_ground_truth)
        assert isinstance(result, float)
