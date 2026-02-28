"""Tests for GPU telemetry monitor summary behavior."""

from workers.gpu_monitor import GPUMonitor


class TestGpuMonitorSummary:
    def test_empty_summary(self):
        mon = GPUMonitor()
        out = mon.summary()
        assert out["gpu_monitor_sample_count"] == 0
        assert out["gpu_util_avg_pct"] is None

    def test_summary_stats(self):
        mon = GPUMonitor()
        mon._samples = [
            {
                "gpu_util_pct": 10.0,
                "gpu_mem_util_pct": 20.0,
                "gpu_mem_used_mb": 1000.0,
                "gpu_mem_total_mb": 16000.0,
                "gpu_mem_used_pct": 6.25,
                "gpu_power_w": 50.0,
                "gpu_temp_c": 45.0,
            },
            {
                "gpu_util_pct": 90.0,
                "gpu_mem_util_pct": 70.0,
                "gpu_mem_used_mb": 2000.0,
                "gpu_mem_total_mb": 16000.0,
                "gpu_mem_used_pct": 12.5,
                "gpu_power_w": 120.0,
                "gpu_temp_c": 65.0,
            },
        ]
        out = mon.summary()
        assert out["gpu_monitor_sample_count"] == 2
        assert out["gpu_util_avg_pct"] == 50.0
        assert out["gpu_util_peak_pct"] == 90.0
        assert out["gpu_mem_used_peak_mb"] == 2000.0
        assert out["gpu_temp_peak_c"] == 65.0
