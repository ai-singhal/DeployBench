MODAL_T4_HOURLY_RATE = 0.59  # $/hr for T4 on Modal


def estimate_cost(avg_latency_ms: float, requests_per_day: int) -> float:
    """
    Given average latency and daily request volume, estimate monthly GPU cost.

    Args:
        avg_latency_ms: Average inference latency in milliseconds.
        requests_per_day: Number of inference requests served per day.

    Returns:
        Estimated monthly GPU cost in USD (rounded to 2 decimal places).
    """
    seconds_per_request = avg_latency_ms / 1000
    total_gpu_seconds_per_day = seconds_per_request * requests_per_day
    gpu_hours_per_day = total_gpu_seconds_per_day / 3600
    monthly_gpu_hours = gpu_hours_per_day * 30
    return round(monthly_gpu_hours * MODAL_T4_HOURLY_RATE, 2)
