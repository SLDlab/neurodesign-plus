# Getting Started

## Public classes

`neurodesign-plus` 2.0 exposes one authoritative implementation of:

```python
from neurodesign import Experiment, Design, Optimisation
```

The compatibility import path remains:

```python
from neurodesign.classes import Experiment, Design, Optimisation
```

Both resolve to the same class objects implemented in `neurodesign.classes`.

## Timing model

A run contains conceptual trials. A conceptual trial contains one or more modeled events.

Flat `order` inputs are shorthand for one-event conceptual trials.

Version 2.0 separates four interval roles:

- `trial_start_interval`
- `post_event_interval`
- `event_transition_interval`
- `inter_trial_interval`

Optional rests are configured with:

- `rest_every_n_trials`
- `rest_interval`

Within-trial `event_transition_interval` values never cross trial boundaries.
`inter_trial_interval` applies only between conceptual trials.
Rests are boundary intervals, not events.

Only modeled event durations enter `Xnonconv`. Displayed fixation or baseline time during an interval is not automatically modeled as task activity.

## Minimal flat one-event example

```python
from neurodesign import Experiment

exp = Experiment(
    TR=2.0,
    n_trials=20,
    P=[0.5, 0.5],
    C=[[1, -1]],
    rho=0.3,
    n_stimuli=2,
    event_durations=1.0,
    trial_start_interval=0.5,
    post_event_interval=0.2,
    inter_trial_interval={"model": "exponential", "mean": 2.0, "min": 1.0, "max": 4.0},
    resolution=0.1,
    seed=42,
)

design = exp.create_design(seed=42)
design.designmatrix()
```

## Fixed multi-event example

```python
trial_templates = [
    {
        "template_id": "standard",
        "trial_type": "standard",
        "events": [
            {"category": "cue_easy", "code": 0, "duration": 0.8},
            {"category": "choice_left", "code": 1, "duration": 1.2},
            {"category": "feedback", "code": 2, "duration": 1.0},
        ],
    },
    {
        "template_id": "hint_branch",
        "trial_type": "hint",
        "events": [
            {"category": "cue_hard", "code": 3, "duration": 0.8},
            {"category": "hint", "code": 4, "duration": 0.7},
            {"category": "choice_right", "code": 5, "duration": 1.2},
            {"category": "feedback", "code": 2, "duration": 1.0},
        ],
    },
]

exp = Experiment(
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
```

## Probabilistic complete-template sampling

```python
exp = Experiment(
    TR=1.0,
    P=[0.18, 0.12, 0.16, 0.12, 0.10, 0.10, 0.22],
    C=[[1, 0, 0, 0, 0, 0, -1]],
    rho=0.3,
    n_stimuli=7,
    trial_templates=[...],
    trial_template_probabilities=[0.4, 0.35, 0.25],
    n_conceptual_trials=10,
    event_durations={...},
    trial_start_interval=0.0,
    post_event_interval=0.0,
    event_transition_interval={...},
    inter_trial_interval={"model": "uniform", "min": 1.0, "mean": 1.45, "max": 1.9},
    seed=12,
)
```

Version 2.0 never truncates a template to satisfy an event budget.

## Requested, normalized, and realized timing

Every timing component preserves three layers:

1. the requested public specification
2. the normalized internal rule representation
3. the realized occurrence-level values on a concrete `Design`

Representative access points:

- `Experiment.export_specification()`
- `Design.export_payload()`
- `Design.realized_event_durations`
- `Design.realized_trial_start_intervals`
- `Design.realized_post_event_intervals`
- `Design.realized_event_transition_intervals`
- `Design.realized_inter_trial_intervals`
- `Design.realized_rest_intervals`

Requested timing specifications are distinct from normalized internal rules and from realized occurrence-level values on a sampled `Design`.

## Selector scope

- `trial_start_interval`: global rule or `by_trial_type`
- `post_event_interval`: global rule or `by_event_category`
- `event_transition_interval`: global rule or `by_event_transition`
- `inter_trial_interval`: global rule only
- `rest_interval`: global rule only

Use `"default"` only inside selector wrappers that support it.

## Duration grammar

A scalar `x` normalizes to:

```python
{"model": "fixed", "value": x}
```

Supported models:

- `fixed`
- `uniform`
- `exponential`
- `gaussian`

Bounded exponential and Gaussian rules use true truncated distributions. For bounded rules, `mean` refers to the mean of the realized bounded distribution.

Supported public rule shapes:

```python
{"model": "fixed", "value": value}
{"model": "uniform", "min": a, "max": b}
{"model": "uniform", "min": a, "mean": (a + b) / 2, "max": b}
{"model": "exponential", "mean": m}
{"model": "exponential", "mean": m, "min": a, "max": b}
{"model": "gaussian", "mean": m, "std": s, "min": a, "max": b}
```

Occurrence-level sampling happens when a `Design` is realized, not when the `Experiment` is declared.
Invalid bounds, impossible bounded means, non-finite values, and uncovered selectors raise validation errors.

## Counts and scoring

Version 2.0 keeps conceptual-trial counts separate from flattened event counts.

- `n_conceptual_trials` governs trial sampling and boundary placement.
- `n_events` governs flattened event-level quantities such as `Ff` and `Fc`.
- `Fe` and `Fd` depend on realized design matrices, so they can change when timing realizations change.

The quickest way to inspect the split is:

```python
spec = exp.export_specification()
payload = design.export_payload()
spec["n_conceptual_trials"], payload["counts"]["n_events"]
```

## Convergence

`Optimisation(convergence=k)` stops when the generation-best score fails to improve for `k` consecutive completed generations.
Set `convergence=None` or `convergence=0` to disable early stopping and run the full planned cycle count.
Equality counts as no improvement, and the current implementation has no minimum-delta tolerance.
Early stopping does not prove that a global optimum was found.

Inspect the optimization state with:

- `optimisation.generations_completed`
- `optimisation.stop_reason`
- `optimisation.optima`

## Recommended workflow

Use this public end-to-end sequence when optimisation is needed:

```python
from pathlib import Path
import json

from neurodesign import Experiment, Optimisation, report

exp = Experiment(...)
opt = Optimisation(
    experiment=exp,
    weights=[0.0, 0.5, 0.25, 0.25],  # order: Fe, Fd, Ff, Fc
    preruncycles=1,
    cycles=2,
    seed=101,
    optimisation="simulation",
    G=3,
    I=1,
    outdes=1,
    convergence=1,
)
opt.optimise()
design = opt.selected_design(0)

report.make_report(opt, Path("report.pdf"))
Path("schedule.json").write_text(
    json.dumps(design.export_payload(), indent=2), encoding="utf-8"
)
Path("specification.json").write_text(
    json.dumps(exp.export_specification(), indent=2), encoding="utf-8"
)
```

Workflow checklist:

1. define the `Experiment`
2. choose one construction mode: flat `order`, fixed `trial_templates` plus `trials`, or probabilistic `trial_templates` plus `trial_template_probabilities` and `n_conceptual_trials`
3. define event-duration and interval rules
4. instantiate `Optimisation` only when search is needed
5. call `optimise()`
6. inspect `generations_completed`, `stop_reason`, and `optima`
7. call `selected_design(0)`
8. inspect conceptual-trial metadata, event metadata, and realized timing
9. generate the report
10. export the schedule and specification
11. save the seed and optimisation settings with the exported files

For fixed or manual designs with no optimisation step, use the no-search route:

```python
design = exp.create_design(seed=12)
# or, for a manually specified flat schedule:
design = exp.create_manual_design(order=[0, 1, 0, 1], inter_trial_intervals=[2.0] * 4)
```

Do not rely on internal attributes such as `.bestdesign`, raw `designs` indexing, or undocumented schedule internals in user workflows.

## Reproducibility

Version 2.0 reproducibility is controlled by the package seed and NumPy `Generator` architecture, not by NumPy global seeding.

With the same complete specification and seed, independent runs reproduce:

- sampled trial-template sequences
- realized event durations and interval values
- selected designs
- scores
- exported schedule payloads

Rendered PDFs and PNGs are not guaranteed to be byte-identical unless a specific artifact path documents that guarantee.

## Next steps

- Read the [migration guide](migration.md) for old-to-new API mapping
- Review the [metrics guide](metrics.md)
- Browse the executed [tutorial notebooks](tutorials.md)
- Use the [API reference](api_reference.md) for class and method signatures
