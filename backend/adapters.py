"""Model adapter detection for uploaded checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AdapterCheck:
    compatible: bool
    adapter: str | None
    task: str | None
    reason: str | None = None


_SUPPORTED_TASKS = {
    "detect": "ultralytics_detect",
    "classify": "ultralytics_classify",
    "segment": "ultralytics_segment",
}



def detect_adapter(model_path: str, task_hint: str | None = None) -> AdapterCheck:
    p = Path(model_path)
    if not p.exists():
        return AdapterCheck(False, None, None, reason=f"Model file does not exist: {model_path}")
    if p.suffix != ".pt":
        return AdapterCheck(False, None, None, reason="Only .pt files are supported")

    try:
        from ultralytics import YOLO

        model = YOLO(str(p))
        task = getattr(model, "task", None)
        if not task and task_hint:
            task = task_hint
        if task not in _SUPPORTED_TASKS:
            return AdapterCheck(
                False,
                None,
                task,
                reason=(
                    f"Unsupported model task '{task}'. Supported tasks: detect, classify, segment. "
                    "If this is a custom torch model, wrap it in Ultralytics YOLO format."
                ),
            )
        return AdapterCheck(True, _SUPPORTED_TASKS[task], task)
    except Exception as exc:  # noqa: BLE001
        return AdapterCheck(
            False,
            None,
            None,
            reason=(
                "Could not load model as an Ultralytics checkpoint. "
                f"Error: {exc}. If your model is a raw torch module, convert/export to an Ultralytics-compatible .pt."
            ),
        )
