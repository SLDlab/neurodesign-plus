import json
import os
from pathlib import Path

import pytest

from neurodesign import Experiment

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "tests" / "fixtures" / "v1_reference"
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".tmp_mpl"))
os.environ.setdefault("IPYTHONDIR", str(REPO_ROOT / ".tmp_ipython"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


@pytest.fixture
def scalar_experiment():
    return Experiment(
        TR=2.0,
        n_trials=8,
        P=[0.5, 0.5],
        C=[[1, -1]],
        n_stimuli=2,
        rho=0.3,
        event_durations=1.0,
        trial_start_interval=0.5,
        post_event_interval=0.2,
        inter_trial_interval=2.0,
        resolution=0.1,
        seed=7,
    )


@pytest.fixture
def multi_event_templates():
    return [
        {
            "template_id": "standard",
            "trial_type": "standard",
            "events": [
                {"category": "cue_easy", "duration": 0.8},
                {"category": "choice_left", "duration": 1.2},
                {"category": "feedback", "duration": 1.0},
            ],
        },
        {
            "template_id": "hint_branch",
            "trial_type": "hint",
            "events": [
                {"category": "cue_hard", "duration": 0.8},
                {"category": "hint", "duration": 0.7},
                {"category": "choice_right", "duration": 1.2},
                {"category": "feedback", "duration": 1.0},
            ],
        },
        {
            "template_id": "hold_branch",
            "trial_type": "hold",
            "events": [
                {"category": "cue_easy", "duration": 0.8},
                {"category": "hold", "duration": 0.9},
                {"category": "choice_left", "duration": 1.2},
                {"category": "feedback", "duration": 1.0},
            ],
        },
    ]


@pytest.fixture
def case10_experiment(multi_event_templates):
    return Experiment(
        TR=1.0,
        P=[0.18, 0.12, 0.16, 0.12, 0.10, 0.10, 0.22],
        C=[
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
        rho=0.3,
        n_stimuli=7,
        trial_templates=multi_event_templates,
        trial_template_probabilities=[0.4, 0.35, 0.25],
        n_conceptual_trials=10,
        trial_start_interval=0.0,
        post_event_interval=0.0,
        event_transition_interval={
            "by_event_transition": {
                ("cue_easy", "choice_left"): {"model": "uniform", "min": 0.3, "max": 1.0},
                ("cue_hard", "hint"): {"model": "uniform", "min": 0.3, "max": 0.9},
                ("hint", "choice_right"): {"model": "uniform", "min": 0.2, "max": 0.8},
                ("cue_easy", "hold"): {"model": "uniform", "min": 0.2, "max": 0.8},
                ("hold", "choice_left"): {"model": "uniform", "min": 0.3, "max": 0.9},
                ("choice_left", "feedback"): {"model": "uniform", "min": 0.2, "max": 0.7},
                ("choice_right", "feedback"): {
                    "model": "uniform",
                    "min": 0.2,
                    "max": 0.7,
                },
            }
        },
        inter_trial_interval={"model": "uniform", "min": 1.0, "mean": 1.45, "max": 1.9},
        event_durations={
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
        resolution=0.1,
        seed=12,
        trial_max=2.2,
    )


@pytest.fixture
def baseline_case10_tutorial():
    return json.loads((BASELINE_DIR / "case10_tutorial_metadata.json").read_text())


@pytest.fixture
def baseline_case10_manuscript():
    return json.loads(
        (BASELINE_DIR / "case10_manuscript_support_metadata.json").read_text()
    )
