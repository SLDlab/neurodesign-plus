from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(
    os.environ.get(
        "NEURODESIGN_VALIDATION_DETERMINISM_OUTPUT",
        REPO_ROOT / "validation" / "_artifacts" / "case10_determinism",
    )
)


def _run_worker(output_path: Path) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "validation.helpers.case10_determinism_worker",
            "--output",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip()
            or completed.stdout.strip()
            or "determinism worker failed"
        )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _arrays_equal(left, right) -> bool:
    return np.array_equal(np.asarray(left), np.asarray(right))


def main() -> int:
    """Run the Case 10 determinism check and save the comparison artifact."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    first_path = OUTPUT_DIR / "run_1.json"
    second_path = OUTPUT_DIR / "run_2.json"
    run_1 = _run_worker(first_path)
    run_2 = _run_worker(second_path)

    exact_fields = [
        "template_sequence",
        "trial_ids",
        "event_order",
        "event_categories",
        "realized_event_durations",
        "realized_trial_start_intervals",
        "realized_post_event_intervals",
        "realized_event_transition_intervals",
        "realized_inter_trial_intervals",
        "realized_rest_intervals",
        "event_onsets",
        "event_offsets",
        "Xnonconv",
        "Xconv",
    ]
    comparisons = {}
    for field in exact_fields:
        comparisons[field] = {
            "equal": _arrays_equal(run_1[field], run_2[field]),
            "tolerance": 0.0,
        }

    metric_fields = ["Fe", "Fd", "Ff", "Fc", "F"]
    for field in metric_fields:
        equal = run_1["metrics"][field] == run_2["metrics"][field]
        comparisons[field] = {"equal": equal, "tolerance": 0.0}

    comparisons["selected_design_hash"] = {
        "equal": run_1["selected_design_hash"] == run_2["selected_design_hash"],
        "tolerance": 0.0,
    }
    comparisons["export_payload"] = {
        "equal": json.dumps(
            run_1["export_payload"], sort_keys=True, separators=(",", ":")
        )
        == json.dumps(run_2["export_payload"], sort_keys=True, separators=(",", ":")),
        "tolerance": 0.0,
    }
    comparisons["specification_payload"] = {
        "equal": json.dumps(
            run_1["specification_payload"], sort_keys=True, separators=(",", ":")
        )
        == json.dumps(
            run_2["specification_payload"], sort_keys=True, separators=(",", ":")
        ),
        "tolerance": 0.0,
    }

    differing = [field for field, result in comparisons.items() if not result["equal"]]
    summary = {
        "run_1": str(first_path),
        "run_2": str(second_path),
        "comparisons": comparisons,
        "differing_fields": differing,
        "all_equal": not differing,
    }
    (OUTPUT_DIR / "comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0 if not differing else 1


if __name__ == "__main__":
    raise SystemExit(main())
