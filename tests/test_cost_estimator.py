"""Tests for utils/cost_estimator.py — estimate_cost() pure function."""
import pytest
from utils.cost_estimator import estimate_cost, MODAL_T4_HOURLY_RATE


class TestModalT4HourlyRate:
    def test_rate_is_correct(self):
        assert MODAL_T4_HOURLY_RATE == 0.59

    def test_rate_is_float(self):
        assert isinstance(MODAL_T4_HOURLY_RATE, float)

    def test_rate_is_positive(self):
        assert MODAL_T4_HOURLY_RATE > 0


class TestEstimateCostFormula:
    """Verify the formula: cost = (latency_ms/1000 * reqs/day / 3600 * 30) * rate"""

    @pytest.mark.parametrize("avg_latency_ms,requests_per_day", [
        (10.0, 1_000),
        (50.0, 5_000),
        (100.0, 10_000),
        (200.0, 50_000),
        (500.0, 100_000),
        (1.0, 1),
        (999.0, 999),
    ])
    def test_formula_matches_spec(self, avg_latency_ms, requests_per_day):
        """Result must exactly match the spec formula, rounded to 2 decimal places."""
        seconds_per_request = avg_latency_ms / 1000
        total_gpu_seconds_per_day = seconds_per_request * requests_per_day
        gpu_hours_per_day = total_gpu_seconds_per_day / 3600
        monthly_gpu_hours = gpu_hours_per_day * 30
        expected = round(monthly_gpu_hours * MODAL_T4_HOURLY_RATE, 2)

        assert estimate_cost(avg_latency_ms, requests_per_day) == expected

    def test_spot_check_100ms_10k_requests(self):
        """Hand-computed sanity check for 100ms @ 10_000 req/day.

        100ms/req * 10_000 req/day = 1_000 GPU-sec/day
        1_000 / 3600 = 0.2778 GPU-hr/day
        0.2778 * 30 = 8.333 GPU-hr/month
        8.333 * 0.59 = $4.92
        """
        assert estimate_cost(100.0, 10_000) == 4.92

    def test_spot_check_50ms_5k_requests(self):
        """50ms @ 5_000 req/day should cost one quarter of 100ms @ 10_000 req/day."""
        cost_half = estimate_cost(50.0, 5_000)
        cost_full = estimate_cost(100.0, 10_000)
        assert cost_half == pytest.approx(cost_full / 4, abs=0.01)


class TestEstimateCostReturnType:
    def test_returns_float(self):
        result = estimate_cost(100.0, 10_000)
        assert isinstance(result, float)

    def test_rounded_to_two_decimal_places(self):
        result = estimate_cost(100.0, 10_000)
        assert result == round(result, 2)

    def test_result_is_non_negative(self):
        assert estimate_cost(100.0, 10_000) >= 0
        assert estimate_cost(0.0, 0) >= 0


class TestEstimateCostEdgeCases:
    def test_zero_requests_returns_zero(self):
        assert estimate_cost(100.0, 0) == 0.0

    def test_zero_latency_returns_zero(self):
        assert estimate_cost(0.0, 10_000) == 0.0

    def test_both_zero_returns_zero(self):
        assert estimate_cost(0.0, 0) == 0.0

    def test_very_small_latency(self):
        """Sub-millisecond latency should still produce a valid cost."""
        result = estimate_cost(0.1, 10_000)
        assert result >= 0.0
        assert isinstance(result, float)

    def test_very_large_requests(self):
        """1 million requests/day with 100ms latency — should not overflow."""
        result = estimate_cost(100.0, 1_000_000)
        expected = round(
            (100 / 1000 * 1_000_000 / 3600 * 30) * MODAL_T4_HOURLY_RATE, 2
        )
        assert result == expected

    def test_very_large_latency(self):
        """Extreme latency (60s) should still compute correctly."""
        result = estimate_cost(60_000.0, 10_000)
        expected = round(
            (60_000 / 1000 * 10_000 / 3600 * 30) * MODAL_T4_HOURLY_RATE, 2
        )
        assert result == expected

    def test_single_request_per_day(self):
        result = estimate_cost(100.0, 1)
        expected = round(
            (100 / 1000 * 1 / 3600 * 30) * MODAL_T4_HOURLY_RATE, 2
        )
        assert result == expected


class TestEstimateCostMonotonicity:
    """Higher latency or more requests should cost more (or equal)."""

    def test_higher_latency_costs_more(self):
        cost_low = estimate_cost(50.0, 10_000)
        cost_high = estimate_cost(100.0, 10_000)
        assert cost_high > cost_low

    def test_more_requests_costs_more(self):
        cost_low = estimate_cost(100.0, 5_000)
        cost_high = estimate_cost(100.0, 10_000)
        assert cost_high > cost_low

    def test_doubling_latency_doubles_cost(self):
        cost_base = estimate_cost(50.0, 10_000)
        cost_double = estimate_cost(100.0, 10_000)
        assert abs(cost_double - 2 * cost_base) < 0.01

    def test_doubling_requests_doubles_cost(self):
        cost_base = estimate_cost(100.0, 5_000)
        cost_double = estimate_cost(100.0, 10_000)
        assert abs(cost_double - 2 * cost_base) < 0.01

    @pytest.mark.parametrize("latency_ms", [10.0, 50.0, 100.0, 200.0])
    def test_costs_are_strictly_ordered_by_latency(self, latency_ms):
        """Any latency increase with fixed requests must strictly increase cost."""
        cost_before = estimate_cost(latency_ms, 10_000)
        cost_after = estimate_cost(latency_ms + 1.0, 10_000)
        assert cost_after > cost_before
