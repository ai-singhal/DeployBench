"""Sandbox process entrypoint for upload lane execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from workers.pipeline import LaneContext, LaneUnsupportedError, normalize_lane_error, run_lane


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a DeployBench lane inside Modal Sandbox")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-ref", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--requests-per-day", type=int, required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ctx = LaneContext(
        config=args.config,
        mode="upload",
        model_ref=args.model_ref,
        task=args.task,
        requests_per_day=args.requests_per_day,
    )
    try:
        payload = run_lane(ctx)
    except LaneUnsupportedError as exc:
        payload = normalize_lane_error(args.config, exc, status="unsupported")
    except Exception as exc:  # noqa: BLE001
        payload = normalize_lane_error(args.config, exc, status="failed")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
