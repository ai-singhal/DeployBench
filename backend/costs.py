"""Cost estimation helpers."""

from __future__ import annotations

from utils.cost_estimator import MODAL_T4_HOURLY_RATE



def estimate_monthly_cost(avg_latency_ms: float, requests_per_day: int, overhead_factor: float = 1.15) -> float:
    seconds_per_request = avg_latency_ms / 1000.0
    gpu_seconds_day = requests_per_day * seconds_per_request
    gpu_hours_month = (gpu_seconds_day / 3600.0) * 30.0
    return round(gpu_hours_month * MODAL_T4_HOURLY_RATE * overhead_factor, 2)
