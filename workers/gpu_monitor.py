"""GPU telemetry sampling utilities for benchmark lanes."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from typing import Any


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
    return ordered[idx]


class GPUMonitor:
    """Sample GPU utilization from nvidia-smi in a background thread."""

    def __init__(self, poll_interval_sec: float = 0.2):
        self.poll_interval_sec = max(0.05, float(poll_interval_sec))
        self._samples: list[dict[str, float]] = []
        self._errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._source = "nvidia-smi"

    def _read_nvidia_smi(self) -> dict[str, float] | None:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=2.0, check=False)
        if proc.returncode != 0:
            msg = proc.stderr.strip() or proc.stdout.strip() or "nvidia-smi failed"
            self._errors.append(msg[:240])
            return None

        line = (proc.stdout or "").strip().splitlines()
        if not line:
            self._errors.append("nvidia-smi returned no output")
            return None

        # If multiple GPUs are present, aggregate by max util and max memory use.
        rows: list[list[float]] = []
        for row in line:
            parts = [p.strip() for p in row.split(",")]
            if len(parts) < 6:
                continue
            try:
                rows.append([float(p) for p in parts[:6]])
            except ValueError:
                continue

        if not rows:
            self._errors.append("Could not parse nvidia-smi metrics")
            return None

        gpu_util = max(r[0] for r in rows)
        mem_util = max(r[1] for r in rows)
        mem_used = max(r[2] for r in rows)
        mem_total = max(r[3] for r in rows)
        power = max(r[4] for r in rows)
        temp = max(r[5] for r in rows)

        mem_used_pct = (mem_used / mem_total) * 100.0 if mem_total > 0 else 0.0

        return {
            "gpu_util_pct": gpu_util,
            "gpu_mem_util_pct": mem_util,
            "gpu_mem_used_mb": mem_used,
            "gpu_mem_total_mb": mem_total,
            "gpu_mem_used_pct": mem_used_pct,
            "gpu_power_w": power,
            "gpu_temp_c": temp,
        }

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = self._read_nvidia_smi()
            if sample is not None:
                self._samples.append(sample)
            self._stop.wait(self.poll_interval_sec)

    def start(self) -> None:
        if shutil.which("nvidia-smi") is None:
            self._source = "unavailable"
            self._errors.append("nvidia-smi not found")
            return

        self._thread = threading.Thread(target=self._loop, name="gpu-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def summary(self) -> dict[str, Any]:
        if not self._samples:
            return {
                "gpu_monitor_source": self._source,
                "gpu_monitor_sample_count": 0,
                "gpu_util_avg_pct": None,
                "gpu_util_p95_pct": None,
                "gpu_util_peak_pct": None,
                "gpu_mem_util_avg_pct": None,
                "gpu_mem_used_peak_mb": None,
                "gpu_mem_used_peak_pct": None,
                "gpu_power_avg_w": None,
                "gpu_temp_peak_c": None,
                "gpu_monitor_errors": list(dict.fromkeys(self._errors))[:3],
            }

        util = [s["gpu_util_pct"] for s in self._samples]
        mem_util = [s["gpu_mem_util_pct"] for s in self._samples]
        mem_used = [s["gpu_mem_used_mb"] for s in self._samples]
        mem_used_pct = [s["gpu_mem_used_pct"] for s in self._samples]
        power = [s["gpu_power_w"] for s in self._samples]
        temp = [s["gpu_temp_c"] for s in self._samples]

        return {
            "gpu_monitor_source": self._source,
            "gpu_monitor_sample_count": len(self._samples),
            "gpu_util_avg_pct": round(sum(util) / len(util), 2),
            "gpu_util_p95_pct": round(_p95(util), 2),
            "gpu_util_peak_pct": round(max(util), 2),
            "gpu_mem_util_avg_pct": round(sum(mem_util) / len(mem_util), 2),
            "gpu_mem_used_peak_mb": round(max(mem_used), 2),
            "gpu_mem_used_peak_pct": round(max(mem_used_pct), 2),
            "gpu_power_avg_w": round(sum(power) / len(power), 2),
            "gpu_temp_peak_c": round(max(temp), 2),
            "gpu_monitor_errors": list(dict.fromkeys(self._errors))[:3],
        }
