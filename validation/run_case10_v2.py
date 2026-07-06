from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from neurodesign import Optimisation
from validation.helpers.case10_v2 import (
    COMMON_SPEC,
    TRIAL_TEMPLATES,
    build_case10_experiment,
)
from validation.helpers.version_metadata import capture_version_metadata

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(
    os.environ.get(
        "NEURODESIGN_VALIDATION_CASE10_OUTPUT",
        REPO_ROOT / "validation" / "_artifacts" / "case10_comparison",
    )
)
BASELINE_DIR = REPO_ROOT / "tests" / "fixtures" / "v1_reference"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(design):
    return {
        "F": float(design.F),
        "Fe": float(design.Fe),
        "Fd": float(design.Fd),
        "Ff": float(design.Ff),
        "Fc": float(design.Fc),
    }


def _json_safe(value):
    if isinstance(value, dict):
        converted = {}
        for key, item in value.items():
            if isinstance(key, tuple):
                key = str(key)
            converted[str(key)] = _json_safe(item)
        return converted
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _design_payload(design):
    return {
        "counts": {
            "n_conceptual_trials": int(design.experiment.n_conceptual_trials),
            "n_events": int(len(design.order)),
        },
        "template_ids": design.trial_template_ids,
        "trial_type_ids": design.trial_type_ids,
        "trial_ids": design.trial_ids.tolist(),
        "event_index_within_trial": design.event_index_within_trial.tolist(),
        "order": design.order,
        "event_categories": design.event_categories,
        "event_onsets": design.event_onsets.tolist(),
        "event_offsets": design.event_offsets.tolist(),
        "trial_starts": design.trial_starts.tolist(),
        "trial_ends": design.trial_ends.tolist(),
        "realized_event_durations": design.realized_event_durations.tolist(),
        "realized_trial_start_intervals": design.realized_trial_start_intervals.tolist(),
        "realized_post_event_intervals": design.realized_post_event_intervals.tolist(),
        "realized_event_transition_intervals": design.realized_event_transition_intervals.tolist(),
        "realized_inter_trial_intervals": design.realized_inter_trial_intervals.tolist(),
        "realized_rest_intervals": design.realized_rest_intervals.tolist(),
        "xnonconv_shape": list(np.asarray(design.Xnonconv).shape),
        "xconv_shape": list(np.asarray(design.Xconv).shape),
        "xnonconv_sum": float(np.asarray(design.Xnonconv).sum()),
        "xconv_sum": float(np.asarray(design.Xconv).sum()),
        "metrics": _metrics(design),
    }


def main() -> None:
    """Generate the version-1 versus version-2 Case 10 comparison artifact."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_tutorial = _load(BASELINE_DIR / "case10_tutorial_metadata.json")
    baseline_manuscript = _load(BASELINE_DIR / "case10_manuscript_support_metadata.json")

    tutorial_trials = [
        {"template_id": TRIAL_TEMPLATES[idx]["template_id"]}
        for idx in [0, 2, 0, 0, 0, 0, 1, 0, 2, 2]
    ]
    tutorial_exp = build_case10_experiment(seed=12, trials=tutorial_trials)
    tutorial_design = tutorial_exp.create_design(seed=12)
    tutorial_design.designmatrix().FCalc(weights=[0.0, 0.5, 0.25, 0.25])

    manuscript_exp = build_case10_experiment(seed=12)
    manuscript_pop = Optimisation(
        experiment=manuscript_exp,
        weights=[0.0, 0.5, 0.25, 0.25],
        preruncycles=1,
        cycles=2,
        seed=20260705,
        optimisation="GA",
        G=4,
        I=2,
        outdes=1,
        folder=OUTPUT_DIR / "download_bundle",
    )
    manuscript_pop.optimise()
    manuscript_pop.evaluate()
    manuscript_pop.download()
    selected = manuscript_pop.selected_design(0)

    payload = {
        "version_metadata": capture_version_metadata(REPO_ROOT),
        "canonical_spec": COMMON_SPEC,
        "template_definitions": TRIAL_TEMPLATES,
        "baseline_paths": {
            "tutorial": str(BASELINE_DIR / "case10_tutorial_metadata.json"),
            "manuscript_support": str(
                BASELINE_DIR / "case10_manuscript_support_metadata.json"
            ),
        },
        "version_1_baseline": {
            "tutorial": {
                "source": baseline_tutorial["source"],
                "sampled_trial_list": baseline_tutorial["sampled_trial_list"],
                "trial_ids": baseline_tutorial["trial_ids"],
                "order": baseline_tutorial["order"],
                "event_onsets": baseline_tutorial["design_onsets"],
                "realized_event_durations": baseline_tutorial[
                    "design_all_stim_durations"
                ],
                "realized_intervals_event_aligned": baseline_tutorial["design_ITI"],
                "metrics": baseline_tutorial["metrics"],
            },
            "manuscript_support": {
                "source": baseline_manuscript["source"],
                "sampled_templates": baseline_manuscript["sampled_templates"],
                "trial_ids": baseline_manuscript["trial_ids"],
                "order": baseline_manuscript["selected_order"],
                "event_onsets": baseline_manuscript["selected_onsets"],
                "realized_event_durations": baseline_manuscript[
                    "selected_all_stim_durations"
                ],
                "realized_intervals_event_aligned": baseline_manuscript["selected_ITI"],
                "metrics": baseline_manuscript["metrics"],
            },
        },
        "version_2": {
            "tutorial": _design_payload(tutorial_design),
            "manuscript_support": _design_payload(selected),
        },
        "differences": {
            "tutorial_event_count_delta": len(tutorial_design.order)
            - len(baseline_tutorial["order"]),
            "manuscript_event_count_delta": len(selected.order)
            - len(baseline_manuscript["selected_order"]),
            "tutorial_metric_deltas": {
                key: float(
                    _metrics(tutorial_design)[key] - baseline_tutorial["metrics"][key]
                )
                for key in ("Fe", "Fd", "Ff", "Fc")
            },
            "manuscript_metric_deltas": {
                key: float(_metrics(selected)[key] - baseline_manuscript["metrics"][key])
                for key in ("Fe", "Fd", "Ff", "Fc")
            },
            "tutorial_xnonconv_sum": float(np.asarray(tutorial_design.Xnonconv).sum()),
            "manuscript_xnonconv_sum": float(np.asarray(selected.Xnonconv).sum()),
            "notes": [
                "Fe and Fd may still differ because version 2.0 models event durations without absorbing surrounding intervals into Xnonconv.",
                "Within-trial event_transition_interval values remain separated from the global inter_trial_interval in version 2.0.",
                "The version-2 RNG uses numpy.random.Generator throughout, so manuscript-support realizations can differ from the Python-random-based baseline.",
                "Ff and Fc are expected to match the version-1 baseline whenever the flattened modeled-event order and P match.",
            ],
        },
    }
    (OUTPUT_DIR / "comparison.json").write_text(
        json.dumps(_json_safe(payload), indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
