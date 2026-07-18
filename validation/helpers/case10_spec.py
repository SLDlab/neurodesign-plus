from __future__ import annotations

from copy import deepcopy

from neurodesign import Experiment

TRIAL_TEMPLATES = [
    {
        "template_id": "standard",
        "trial_type": "standard",
        "events": [
            {"category": "cue_easy", "code": 0, "duration": 0.8},
            {
                "category": "choice_left",
                "code": 2,
                "duration": {"model": "uniform", "min": 1.0, "mean": 1.6, "max": 2.2},
            },
            {"category": "feedback", "code": 6, "duration": 1.0},
        ],
    },
    {
        "template_id": "hint_branch",
        "trial_type": "hint",
        "events": [
            {"category": "cue_hard", "code": 1, "duration": 0.8},
            {"category": "hint", "code": 4, "duration": 0.7},
            {
                "category": "choice_right",
                "code": 3,
                "duration": {"model": "uniform", "min": 1.0, "mean": 1.6, "max": 2.2},
            },
            {"category": "feedback", "code": 6, "duration": 1.0},
        ],
    },
    {
        "template_id": "hold_branch",
        "trial_type": "hold",
        "events": [
            {"category": "cue_easy", "code": 0, "duration": 0.8},
            {"category": "hold", "code": 5, "duration": 0.9},
            {
                "category": "choice_left",
                "code": 2,
                "duration": {"model": "uniform", "min": 1.0, "mean": 1.6, "max": 2.2},
            },
            {"category": "feedback", "code": 6, "duration": 1.0},
        ],
    },
]

EVENT_TRANSITION_INTERVAL = {
    "by_event_transition": {
        ("cue_easy", "choice_left"): {"model": "uniform", "min": 0.3, "max": 1.0},
        ("cue_hard", "hint"): {"model": "uniform", "min": 0.3, "max": 0.9},
        ("hint", "choice_right"): {"model": "uniform", "min": 0.2, "max": 0.8},
        ("cue_easy", "hold"): {"model": "uniform", "min": 0.2, "max": 0.8},
        ("hold", "choice_left"): {"model": "uniform", "min": 0.3, "max": 0.9},
        ("choice_left", "feedback"): {"model": "uniform", "min": 0.2, "max": 0.7},
        ("choice_right", "feedback"): {"model": "uniform", "min": 0.2, "max": 0.7},
    }
}

INTER_TRIAL_INTERVAL = {"model": "uniform", "min": 1.0, "mean": 1.45, "max": 1.9}

COMMON_SPEC = {
    "TR": 1.0,
    "P": [0.18, 0.12, 0.16, 0.12, 0.10, 0.10, 0.22],
    "C": [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, -1, 0, 0, 0],
        [1, -1, 0, 0, 0, 0, 0],
    ],
    "rho": 0.3,
    "n_stimuli": 7,
    "resolution": 0.1,
    "trial_templates": TRIAL_TEMPLATES,
    "trial_start_interval": 0.0,
    "post_event_interval": 0.0,
    "event_transition_interval": EVENT_TRANSITION_INTERVAL,
    "inter_trial_interval": INTER_TRIAL_INTERVAL,
    "rest_interval": 0.0,
    "event_durations": {
        "by_event_category": {
            "cue_easy": 0.8,
            "cue_hard": 0.8,
            "choice_left": {"model": "uniform", "min": 1.0, "mean": 1.6, "max": 2.2},
            "choice_right": {"model": "uniform", "min": 1.0, "mean": 1.6, "max": 2.2},
            "hint": 0.7,
            "hold": 0.9,
            "feedback": 1.0,
        }
    },
    "trial_max": 2.2,
}


def build_case10_experiment(
    *,
    seed: int = 12,
    trials: list[dict[str, str]] | None = None,
    trial_template_probabilities: list[float] | None = None,
    n_conceptual_trials: int | None = None,
) -> Experiment:
    """Build the shared Case 10 experiment specification."""
    spec = deepcopy(COMMON_SPEC)
    spec["seed"] = seed
    if trials is not None:
        spec["trials"] = trials
    else:
        spec["trial_template_probabilities"] = (
            [0.4, 0.35, 0.25]
            if trial_template_probabilities is None
            else trial_template_probabilities
        )
        spec["n_conceptual_trials"] = (
            10 if n_conceptual_trials is None else n_conceptual_trials
        )
    return Experiment(**spec)
