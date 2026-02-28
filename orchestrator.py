"""
Orchestrator — pure Python, no Modal imports.

Receives Modal spawn handles from app.py, collects results, computes cost
estimates and a recommendation, and delegates deploy-script generation.
"""

from utils.cost_estimator import estimate_cost
from utils.deploy_script import generate_deploy_script

# Human-readable labels that match what each worker sets in result["config"].
# Used as fallback text when a worker raises before returning its result.
_CONFIG_LABELS: dict[str, str] = {
    "fp32": "FP32 (Baseline)",
    "fp16": "FP16 (Half Precision)",
    "int8": "INT8 (Quantized)",
    "onnx": "ONNX + FP16",
}


def run_all_benchmarks(
    model_name: str,
    requests_per_day: int,
    calls_dict: dict,
) -> dict:
    """
    Collect results from parallel worker spawns and build the final report.

    Args:
        model_name: e.g. "yolov8s"
        requests_per_day: Daily request volume used for cost estimation.
        calls_dict: Mapping of config key → Modal spawn handle.
                    Keys must be a subset of {"fp32", "fp16", "int8", "onnx"}.
                    Each handle's .get() returns a result dict or raises.

    Returns:
        {
            "model": str,
            "results": list[dict],        # one entry per config; errors included
            "recommendation": dict | None, # None when all workers failed
            "deploy_script": str,
        }

    Result dict shape (successful worker):
        {
            "config": str,
            "avg_latency_ms": float,
            "p95_latency_ms": float,
            "fps": float,
            "peak_memory_mb": float,
            "mAP_50": float | <absent>,   # may be missing on some workers
            "est_monthly_cost": float,    # added here
        }

    Result dict shape (failed worker):
        {"config": str, "error": str}
    """
    results: list[dict] = []

    for key, call in calls_dict.items():
        config_label = _CONFIG_LABELS.get(key, key)
        try:
            result = call.get()
            result["est_monthly_cost"] = estimate_cost(
                result["avg_latency_ms"], requests_per_day
            )
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            results.append({"config": config_label, "error": str(exc)})

    valid = [r for r in results if "error" not in r]

    if not valid:
        return {
            "model": model_name,
            "results": results,
            "recommendation": None,
            "deploy_script": "",
        }

    valid.sort(key=lambda r: r["est_monthly_cost"])
    cheapest = valid[0]

    # FP32 is the canonical baseline; fall back to most expensive if it failed.
    baseline = next(
        (r for r in valid if "FP32" in r.get("config", "")),
        valid[-1],
    )

    # Accuracy tradeoff is optional — some workers may omit mAP_50.
    cheapest_map = cheapest.get("mAP_50")
    baseline_map = baseline.get("mAP_50")
    accuracy_tradeoff = (
        round(baseline_map - cheapest_map, 4)
        if cheapest_map is not None and baseline_map is not None
        else None
    )

    recommendation = {
        "best_config": cheapest["config"],
        "monthly_savings": round(
            baseline["est_monthly_cost"] - cheapest["est_monthly_cost"], 2
        ),
        "accuracy_tradeoff": accuracy_tradeoff,
        "speedup": round(baseline["avg_latency_ms"] / cheapest["avg_latency_ms"], 2),
    }

    deploy_script = generate_deploy_script(model_name, cheapest["config"])

    return {
        "model": model_name,
        "results": results,
        "recommendation": recommendation,
        "deploy_script": deploy_script,
    }
