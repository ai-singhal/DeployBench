"""Recommendation engine for lane results."""

from __future__ import annotations

from typing import Any

SAFE_QUALITY_DROP = 0.01



def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None



def compute_agreement(signature_a: dict, signature_b: dict, task: str) -> float:
    """Return [0, 1] agreement between two lightweight prediction signatures."""
    if not signature_a or not signature_b:
        return 0.0

    imgs_a = signature_a.get("images", {})
    imgs_b = signature_b.get("images", {})
    common = sorted(set(imgs_a) & set(imgs_b))
    if not common:
        return 0.0

    scores = []
    for key in common:
        a = imgs_a[key]
        b = imgs_b[key]
        if task == "classify":
            scores.append(1.0 if a.get("top1") == b.get("top1") else 0.0)
        elif task == "segment":
            # Segment path uses class overlap proxy in v1.
            aset = set(a.get("classes", []))
            bset = set(b.get("classes", []))
            if not aset and not bset:
                scores.append(1.0)
            elif aset or bset:
                scores.append(len(aset & bset) / max(1, len(aset | bset)))
        else:
            # Detection: class overlap + count proximity proxy.
            aset = set(a.get("classes", []))
            bset = set(b.get("classes", []))
            overlap = len(aset & bset) / max(1, len(aset | bset)) if aset or bset else 1.0
            count_a = int(a.get("count", 0))
            count_b = int(b.get("count", 0))
            count_term = 1.0 - (abs(count_a - count_b) / max(1, max(count_a, count_b)))
            scores.append(max(0.0, (overlap + count_term) / 2.0))

    return round(sum(scores) / len(scores), 4)



def build_recommendation(lanes: dict[str, dict], task: str) -> dict | None:
    done = [lane for lane in lanes.values() if lane.get("status") == "done"]
    if not done:
        return None

    baseline = next((r for r in done if r.get("config") == "FP32"), None)
    if baseline is None:
        return {
            "best_config": done[0].get("config"),
            "monthly_savings": 0.0,
            "speedup_vs_fp32": None,
            "quality_tradeoff": None,
            "decision_reason": "FP32 baseline unavailable; selected best completed lane by cost.",
            "confidence": "low",
        }

    baseline_cost = _as_float(baseline.get("est_monthly_cost")) or 0.0
    baseline_latency = _as_float(baseline.get("avg_latency_ms"))
    baseline_quality = _as_float(baseline.get("quality_metric"))

    safe_candidates = []
    for lane in done:
        cost = _as_float(lane.get("est_monthly_cost"))
        if cost is None:
            continue

        quality = _as_float(lane.get("quality_metric"))
        if lane.get("config") == "FP32":
            safe_candidates.append(lane)
            continue

        if baseline_quality is None or quality is None:
            # If we cannot quantify quality, only FP32 is considered safe.
            continue

        drop = baseline_quality - quality
        lane["quality_vs_fp32_delta"] = round(drop, 4)
        if drop <= SAFE_QUALITY_DROP:
            safe_candidates.append(lane)

    if safe_candidates:
        chosen = min(safe_candidates, key=lambda r: float(r.get("est_monthly_cost", 1e18)))
        decision_reason = (
            "Selected lowest-cost lane passing quality safety gate (<=1.0% drop vs FP32)."
        )
        confidence = "high" if chosen.get("config") != "FP32" else "medium"
    else:
        chosen = max(done, key=lambda r: float(r.get("quality_metric", 0.0)))
        decision_reason = "No lane passed quality safety gate; selected highest-quality completed lane."
        confidence = "medium"

    chosen_cost = _as_float(chosen.get("est_monthly_cost")) or baseline_cost
    chosen_latency = _as_float(chosen.get("avg_latency_ms"))
    chosen_quality = _as_float(chosen.get("quality_metric"))

    speedup = None
    if baseline_latency and chosen_latency and chosen_latency > 0:
        speedup = round(baseline_latency / chosen_latency, 2)

    quality_tradeoff = None
    if baseline_quality is not None and chosen_quality is not None:
        quality_tradeoff = round(baseline_quality - chosen_quality, 4)

    return {
        "best_config": chosen.get("config"),
        "monthly_savings": round(max(0.0, baseline_cost - chosen_cost), 2),
        "speedup_vs_fp32": speedup,
        "quality_tradeoff": quality_tradeoff,
        "decision_reason": decision_reason,
        "confidence": confidence,
        "quality_metric_name": (
            "mAP50" if task == "detect" else "Top-1 Agreement" if task == "classify" else "Mask/Class Agreement"
        ),
    }
