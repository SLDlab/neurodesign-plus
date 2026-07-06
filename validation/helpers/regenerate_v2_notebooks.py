from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIALS = REPO_ROOT / "tutorials"
NOTEBOOKS = [
    TUTORIALS / "tutorial_1-neurodesign_base_overview.ipynb",
    TUTORIALS / "tutorial_2-comparing_designs_across_experiments.ipynb",
    TUTORIALS / "tutorial_3-progressive_experiment_building.ipynb",
    TUTORIALS / "base_functions" / "tutorial_base-comparing_designs.ipynb",
    TUTORIALS / "base_functions" / "tutorial_base-designing_scoring_and_optimizing.ipynb",
    TUTORIALS / "base_functions" / "tutorial_base-discovering_best_design.ipynb",
    TUTORIALS / "base_functions" / "tutorial_base-optimizing_and_reporting.ipynb",
    TUTORIALS / "new_functions" / "tutorial_new-event_and_trial_intervals.ipynb",
    TUTORIALS / "new_functions" / "tutorial_new-fixed_ordering.ipynb",
    TUTORIALS / "new_functions" / "tutorial_new-probabilistic_ordering.ipynb",
    TUTORIALS / "new_functions" / "tutorial_new-variable_event_durations.ipynb",
]


INTRO = """# {title}

This notebook is part of the neurodesign-plus 2.0 tutorial audit set.

It teaches the current public workflow:

- `Experiment` stores the requested specification.
- `Design` stores one realized schedule with conceptual-trial and event metadata.
- `Optimisation` searches over designs and the authoritative public selection path is `selected_design(rank)`.

Timing is separated into:

- `event_durations`
- `trial_start_interval`
- `post_event_interval`
- `event_transition_interval`
- `inter_trial_interval`
- optional boundary rests via `rest_every_n_trials` and `rest_interval`
"""


SHARED_IMPORTS = """
from pathlib import Path
from copy import deepcopy
import json
import os
import warnings

import numpy as np

from neurodesign import Design, Experiment, Optimisation, report

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".tmp_mpl"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore", message='install "ipywidgets" for Jupyter support')
np.set_printoptions(suppress=True, precision=3)

DEFAULT_WEIGHTS = [0.0, 0.5, 0.25, 0.25]

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


def score_design(design, weights=DEFAULT_WEIGHTS):
    design.designmatrix().FCalc(weights=weights)
    return design


def build_case10_experiment(
    *,
    seed: int = 12,
    trials: list[dict[str, str]] | None = None,
    trial_template_probabilities: list[float] | None = None,
    n_conceptual_trials: int | None = None,
):
    spec = deepcopy(COMMON_SPEC)
    spec["seed"] = seed
    if trials is not None:
        spec["trials"] = trials
    else:
        spec["trial_template_probabilities"] = (
            [0.4, 0.35, 0.25] if trial_template_probabilities is None else trial_template_probabilities
        )
        spec["n_conceptual_trials"] = 10 if n_conceptual_trials is None else n_conceptual_trials
    return Experiment(**spec)
"""


FLAT_EXAMPLE = """
flat_exp = Experiment(
    TR=2.0,
    n_trials=8,
    P=[0.5, 0.5],
    C=[[1, -1]],
    rho=0.3,
    n_stimuli=2,
    event_durations=1.0,
    trial_start_interval=0.5,
    post_event_interval=0.2,
    inter_trial_interval={"model": "exponential", "mean": 2.0, "min": 1.0, "max": 4.0},
    resolution=0.1,
    seed=7,
)
flat_design = score_design(flat_exp.create_design(seed=7))
{
    "mode": flat_exp.mode,
    "n_conceptual_trials": flat_exp.n_conceptual_trials,
    "n_events": len(flat_design.order),
    "requested_inter_trial_interval": flat_exp.export_specification()["inter_trial_interval_requested"],
    "realized_inter_trial_intervals": flat_design.realized_inter_trial_intervals.tolist(),
    "event_index_within_trial": flat_design.event_index_within_trial.tolist(),
}
"""


MANUAL_FLAT_EXAMPLE = """
manual_exp = Experiment(
    TR=2.0,
    n_trials=4,
    P=[0.5, 0.5],
    C=[[1, -1]],
    rho=0.3,
    n_stimuli=2,
    event_durations=1.0,
    trial_start_interval=0.5,
    post_event_interval=0.2,
    inter_trial_interval=2.0,
    resolution=0.1,
    seed=7,
)
manual_design = score_design(
    manual_exp.create_manual_design(
        order=[0, 1, 0, 1],
        inter_trial_intervals=[2.0, 2.0, 2.0, 2.0],
        event_durations=[1.0, 1.0, 1.0, 1.0],
    )
)
{
    "manual_design_type": type(manual_design).__name__,
    "manual_event_onsets": manual_design.event_onsets.tolist(),
    "manual_realized_durations": manual_design.realized_event_durations.tolist(),
    "manual_schedule_head": manual_design.export_schedule()[:2],
}
"""


FIXED_TRIAL_EXAMPLE = """
trial_templates = [
    {
        "template_id": "standard",
        "trial_type": "standard",
        "events": [
            {"category": "cue_easy", "code": 0, "duration": 0.8},
            {"category": "choice_left", "code": 1, "duration": {"model": "uniform", "min": 1.0, "mean": 1.6, "max": 2.2}},
            {"category": "feedback", "code": 2, "duration": 1.0},
        ],
    },
    {
        "template_id": "hint_branch",
        "trial_type": "hint",
        "events": [
            {"category": "cue_hard", "code": 3, "duration": 0.8},
            {"category": "hint", "code": 4, "duration": 0.7},
            {"category": "choice_right", "code": 5, "duration": {"model": "uniform", "min": 1.0, "mean": 1.6, "max": 2.2}},
            {"category": "feedback", "code": 2, "duration": 1.0},
        ],
    },
]
fixed_exp = Experiment(
    TR=1.0,
    P=[1 / 6] * 6,
    C=[[1, 0, 0, 0, 0, -1]],
    rho=0.3,
    n_stimuli=6,
    trial_templates=trial_templates,
    trials=[{"template_id": "standard"}, {"template_id": "hint_branch"}],
    event_durations=1.0,
    trial_start_interval={"by_trial_type": {"standard": 0.5, "hint": 0.7}},
    post_event_interval={"by_event_category": {"feedback": 0.3, "default": 0.1}},
    event_transition_interval={
        "by_event_transition": {
            ("cue_easy", "choice_left"): 0.4,
            ("choice_left", "feedback"): 0.6,
            ("cue_hard", "hint"): 0.2,
            ("hint", "choice_right"): 0.3,
            ("choice_right", "feedback"): 0.5,
        }
    },
    inter_trial_interval=1.5,
    rest_every_n_trials=2,
    rest_interval=3.0,
    resolution=0.1,
    seed=5,
)
fixed_design = score_design(fixed_exp.create_design(seed=5))
{
    "mode": fixed_exp.mode,
    "n_conceptual_trials": fixed_exp.n_conceptual_trials,
    "n_events": len(fixed_design.order),
    "trial_template_ids": fixed_design.trial_template_ids,
    "trial_type_ids": fixed_design.trial_type_ids,
    "schedule_head": fixed_design.export_schedule()[:3],
}
"""


CASE10_EXAMPLE = """
case10_exp = build_case10_experiment(seed=12)
case10_design = score_design(case10_exp.create_design(seed=12))
{
    "n_conceptual_trials": case10_exp.n_conceptual_trials,
    "n_events": len(case10_design.order),
    "trial_template_ids": case10_design.trial_template_ids,
    "trial_starts": case10_design.trial_starts.tolist(),
    "event_index_within_trial": case10_design.event_index_within_trial.tolist(),
    "realized_event_durations": case10_design.realized_event_durations.tolist(),
    "realized_event_transition_intervals": case10_design.realized_event_transition_intervals.tolist(),
    "realized_inter_trial_intervals": case10_design.realized_inter_trial_intervals.tolist(),
    "metrics": {"F": case10_design.F, "Fd": case10_design.Fd, "Ff": case10_design.Ff, "Fc": case10_design.Fc},
}
"""


OPTIMISATION_EXAMPLE = """
case10_opt = Optimisation(
    experiment=build_case10_experiment(seed=12),
    weights=DEFAULT_WEIGHTS,
    preruncycles=1,
    cycles=2,
    seed=101,
    optimisation="simulation",
    G=3,
    I=1,
    outdes=2,
    convergence=1,
    folder=Path("output") / "tutorial_report",
)
case10_opt.optimise()
selected_design = case10_opt.selected_design(0)
{
    "completed_generations": case10_opt.generations_completed,
    "stop_reason": case10_opt.stop_reason,
    "best_score_history": [float(value) for value in case10_opt.optima],
    "selected_rank_0_templates": selected_design.trial_template_ids,
    "selected_rank_0_metrics": {"F": selected_design.F, "Fd": selected_design.Fd, "Ff": selected_design.Ff, "Fc": selected_design.Fc},
    "available_selected_ranks": len(case10_opt.out),
}
"""


CONVERGENCE_EXAMPLE = """
convergence_exp = Experiment(
    TR=2.0,
    n_trials=4,
    P=[0.5, 0.5],
    C=[[1, -1]],
    rho=0.3,
    n_stimuli=2,
    order=[0, 1, 0, 1],
    event_durations=1.0,
    trial_start_interval=0.0,
    post_event_interval=0.0,
    inter_trial_interval=0.0,
    resolution=0.1,
    seed=7,
)
patience_demo = Optimisation(
    experiment=convergence_exp,
    weights=[0.0, 0.0, 0.25, 0.25],
    preruncycles=1,
    cycles=4,
    seed=123,
    optimisation="simulation",
    G=2,
    I=1,
    outdes=1,
    convergence=1,
)
patience_demo.optimise()
patience_design = patience_demo.selected_design(0)

disabled_demo = Optimisation(
    experiment=convergence_exp,
    weights=[0.0, 0.0, 0.25, 0.25],
    preruncycles=1,
    cycles=4,
    seed=123,
    optimisation="simulation",
    G=2,
    I=1,
    outdes=1,
    convergence=None,
)
disabled_demo.optimise()
disabled_design = disabled_demo.selected_design(0)

{
    "patience_demo": {
        "generations_completed": patience_demo.generations_completed,
        "stop_reason": patience_demo.stop_reason,
        "best_score_history": [float(value) for value in patience_demo.optima],
        "selected_design_score": float(patience_design.F),
    },
    "disabled_demo": {
        "generations_completed": disabled_demo.generations_completed,
        "stop_reason": disabled_demo.stop_reason,
        "best_score_history": [float(value) for value in disabled_demo.optima],
        "selected_design_score": float(disabled_design.F),
    },
}
"""


EXPORT_EXAMPLE = """
report_dir = Path("output") / "tutorial_exports"
report_dir.mkdir(parents=True, exist_ok=True)
report_path = report_dir / "report.pdf"
report.make_report(case10_opt, report_path)
payload = selected_design.export_payload()
spec = case10_opt.exp.export_specification()
(report_dir / "schedule.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
(report_dir / "specification.json").write_text(json.dumps(spec, indent=2, default=str), encoding="utf-8")
reloaded = json.loads((report_dir / "schedule.json").read_text(encoding="utf-8"))
{
    "report_path": str(report_path),
    "schedule_path": str(report_dir / "schedule.json"),
    "specification_path": str(report_dir / "specification.json"),
    "reloaded_first_event": reloaded["schedule"][0],
    "reloaded_counts": reloaded["counts"],
}
"""


SELECTION_GUARDRAILS_EXAMPLE = """
pre_optimisation = Optimisation(
    experiment=build_case10_experiment(seed=12),
    weights=DEFAULT_WEIGHTS,
    preruncycles=1,
    cycles=1,
    seed=404,
    optimisation="simulation",
    G=2,
    I=1,
    outdes=1,
)
messages = {}
try:
    pre_optimisation.selected_design(0)
except Exception as exc:
    messages["before_optimise"] = f"{type(exc).__name__}: {exc}"
try:
    case10_opt.selected_design(99)
except Exception as exc:
    messages["invalid_rank"] = f"{type(exc).__name__}: {exc}"
messages
"""


REPRODUCIBILITY_EXAMPLE = """
run_a = Optimisation(
    experiment=build_case10_experiment(seed=12),
    weights=DEFAULT_WEIGHTS,
    preruncycles=1,
    cycles=1,
    seed=808,
    optimisation="simulation",
    G=2,
    I=1,
    outdes=1,
)
run_b = Optimisation(
    experiment=build_case10_experiment(seed=12),
    weights=DEFAULT_WEIGHTS,
    preruncycles=1,
    cycles=1,
    seed=808,
    optimisation="simulation",
    G=2,
    I=1,
    outdes=1,
)
run_a.optimise()
run_b.optimise()
selected_a = run_a.selected_design(0)
selected_b = run_b.selected_design(0)
payload_a = selected_a.export_payload()
payload_b = selected_b.export_payload()
{
    "same_template_sequence": selected_a.trial_template_ids == selected_b.trial_template_ids,
    "same_realized_durations": payload_a["schedule_arrays"]["realized_event_durations"] == payload_b["schedule_arrays"]["realized_event_durations"],
    "same_scores": payload_a["metrics"] == payload_b["metrics"],
    "same_export_payload": payload_a == payload_b,
}
"""


LEGACY_ERROR_EXAMPLE = """
try:
    Experiment(
        TR=2.0,
        n_trials=4,
        P=[0.5, 0.5],
        C=[[1, -1]],
        rho=0.3,
        n_stimuli=2,
        event_durations=1.0,
        t_pre=0.5,
    )
except Exception as exc:
    {"removed_name_error": f"{type(exc).__name__}: {exc}"}
"""


NOTEBOOK_CELL_MAP = {
    "tutorial_1-neurodesign_base_overview.ipynb": [
        (
            "markdown",
            "## Case 1. Flat one-event shorthand\nEach order entry is a one-event conceptual trial, so `n_conceptual_trials` equals `n_events`.",
        ),
        ("code", FLAT_EXAMPLE),
        (
            "markdown",
            "## Case 2. Fixed conceptual trials\nA conceptual trial can contain multiple modeled events, within-trial transitions, and boundary rests.",
        ),
        ("code", FIXED_TRIAL_EXAMPLE),
    ],
    "tutorial_2-comparing_designs_across_experiments.ipynb": [
        (
            "markdown",
            "## Comparison workbook\nThis maintained notebook compares requested and realized timing across a flat design and a fixed multi-event design.",
        ),
        ("code", FLAT_EXAMPLE),
        ("code", FIXED_TRIAL_EXAMPLE),
    ],
    "tutorial_3-progressive_experiment_building.ipynb": [
        (
            "markdown",
            "## Case 10. Canonical integrated workflow\nThis notebook uses the shared Case 10 specification source, runs optimisation, selects rank 0 through the public API, and exports the selected schedule.",
        ),
        ("code", CASE10_EXAMPLE),
        ("code", OPTIMISATION_EXAMPLE),
        ("code", EXPORT_EXAMPLE),
        ("code", REPRODUCIBILITY_EXAMPLE),
    ],
    "tutorial_base-comparing_designs.ipynb": [
        (
            "markdown",
            "## Case 3. Comparing designs\nCompare trial-aware schedules through separate conceptual-trial counts, event counts, realized arrays, and metrics.",
        ),
        ("code", FLAT_EXAMPLE),
        ("code", FIXED_TRIAL_EXAMPLE),
    ],
    "tutorial_base-designing_scoring_and_optimizing.ipynb": [
        (
            "markdown",
            "## Case 4. Building, scoring, and stopping\nScore a fixed design directly, then run two real optimization passes to inspect patience-based early stopping and the disabled-stopping path.\n\n`convergence=k` means `k` consecutive completed generations without strict improvement in the generation-best objective score. Equality counts as no improvement. The current implementation has no minimum-delta tolerance. Early stopping does not prove a global optimum.",
        ),
        ("code", FIXED_TRIAL_EXAMPLE),
        ("code", CONVERGENCE_EXAMPLE),
        ("code", OPTIMISATION_EXAMPLE),
    ],
    "tutorial_base-discovering_best_design.ipynb": [
        (
            "markdown",
            "## Case 5. Authoritative design selection\nRun optimisation, retrieve the selected design with `selected_design(0)`, and inspect selection guardrails.",
        ),
        ("code", OPTIMISATION_EXAMPLE),
        ("code", SELECTION_GUARDRAILS_EXAMPLE),
    ],
    "tutorial_base-optimizing_and_reporting.ipynb": [
        (
            "markdown",
            "## Case 6. Reports and reconstructable exports\nGenerate the report and schedule/specification files from the same selected design workflow.",
        ),
        ("code", OPTIMISATION_EXAMPLE),
        ("code", EXPORT_EXAMPLE),
    ],
    "tutorial_new-event_and_trial_intervals.ipynb": [
        (
            "markdown",
            "## Case 7. Within-trial versus between-trial intervals\n`event_transition_interval` only applies inside a conceptual trial. `inter_trial_interval` only applies between conceptual trials.",
        ),
        ("code", FIXED_TRIAL_EXAMPLE),
        ("code", LEGACY_ERROR_EXAMPLE),
    ],
    "tutorial_new-fixed_ordering.ipynb": [
        (
            "markdown",
            "## Case 8. Fixed conceptual-trial sequences\nUse `trial_templates` plus explicit `trials` when the trial sequence is fixed and no optimisation is required.",
        ),
        ("code", FIXED_TRIAL_EXAMPLE),
        ("code", MANUAL_FLAT_EXAMPLE),
    ],
    "tutorial_new-probabilistic_ordering.ipynb": [
        (
            "markdown",
            "## Case 9. Probabilistic complete-template sampling\nSample complete conceptual trials with `trial_template_probabilities` and `n_conceptual_trials` and then select the reported design through `selected_design(0)`.",
        ),
        ("code", CASE10_EXAMPLE),
        ("code", OPTIMISATION_EXAMPLE),
    ],
    "tutorial_new-variable_event_durations.ipynb": [
        (
            "markdown",
            "## Duration case. Event occupancy versus intervals\nOnly modeled event durations fill `Xnonconv`; trial-start, post-event, transition, inter-trial, and rest intervals remain outside task-event regressors.",
        ),
        ("code", FIXED_TRIAL_EXAMPLE),
        ("code", CASE10_EXAMPLE),
    ],
}


def build_notebook(path: Path) -> nbformat.NotebookNode:
    """Construct one maintained tutorial notebook from the shared cell map."""
    title = path.stem.replace("_", " ")
    nb = new_notebook()
    cells = [
        new_markdown_cell(INTRO.format(title=title)),
        new_code_cell(SHARED_IMPORTS.strip()),
    ]
    for kind, content in NOTEBOOK_CELL_MAP[path.name]:
        if kind == "markdown":
            cells.append(new_markdown_cell(content))
        else:
            cells.append(new_code_cell(content.strip()))
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    return nb


def execute_notebook(path: Path) -> None:
    """Execute and overwrite one maintained tutorial notebook."""
    nb = build_notebook(path)
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()
    path.write_text(nbformat.writes(nb), encoding="utf-8")


def main() -> None:
    """Regenerate and execute the maintained version-2 tutorial notebooks."""
    os.environ.setdefault("PYTHONPATH", str(REPO_ROOT))
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".tmp_mpl"))
    os.environ.setdefault("IPYTHONDIR", str(REPO_ROOT / ".tmp_ipython"))
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    for notebook in NOTEBOOKS:
        notebook.parent.mkdir(parents=True, exist_ok=True)
        execute_notebook(notebook)
        print(notebook.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
