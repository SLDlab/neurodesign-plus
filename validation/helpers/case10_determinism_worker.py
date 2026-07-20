from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neurodesign import Optimisation
from validation.helpers.case10_spec import build_case10_experiment
from validation.helpers.version_metadata import capture_version_metadata


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _metrics(design):
    return {
        "F": float(design.F),
        "Fe": float(design.Fe),
        "Fd": float(design.Fd),
        "Ff": float(design.Ff),
        "Fc": float(design.Fc),
    }


def main() -> int:
    """Generate one deterministic Case 10 payload for comparison runs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle_dir = output.parent / "download_bundle"

    exp = build_case10_experiment(seed=12)
    pop = Optimisation(
        experiment=exp,
        weights=[0.0, 0.5, 0.25, 0.25],
        preruncycles=1,
        cycles=2,
        seed=20260705,
        optimisation="GA",
        G=4,
        I=2,
        outdes=1,
        folder=bundle_dir,
    )
    pop.optimise()
    pop.evaluate()
    pop.download()
    design = pop.selected_design(0)
    export_payload = json.loads(
        (bundle_dir / "design_0" / "event_schedule.json").read_text(encoding="utf-8")
    )
    specification = exp.export_specification()

    payload = {
        "version_metadata": capture_version_metadata(Path(__file__).resolve().parents[2]),
        "optimisation_settings": {
            "weights": pop.weights,
            "preruncycles": pop.preruncycles,
            "cycles": pop.cycles,
            "seed": pop.seed,
            "optimisation": pop.optimisation,
            "G": pop.G,
            "I": pop.I,
            "outdes": pop.outdes,
            "convergence": pop.convergence,
        },
        "selected_design_hash": design.stable_hash(),
        "specification_hash": exp.specification_hash(),
        "template_sequence": design.template_sequence,
        "trial_ids": design.trial_ids,
        "event_order": design.order,
        "event_categories": design.event_categories,
        "realized_event_durations": design.realized_event_durations,
        "realized_trial_start_intervals": design.realized_trial_start_intervals,
        "realized_post_event_intervals": design.realized_post_event_intervals,
        "realized_event_transition_intervals": design.realized_event_transition_intervals,
        "realized_inter_trial_intervals": design.realized_inter_trial_intervals,
        "realized_rest_intervals": design.realized_rest_intervals,
        "event_onsets": design.event_onsets,
        "event_offsets": design.event_offsets,
        "Xnonconv": design.Xnonconv,
        "Xconv": design.Xconv,
        "metrics": _metrics(design),
        "export_payload": export_payload,
        "specification_payload": specification,
    }
    output.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
