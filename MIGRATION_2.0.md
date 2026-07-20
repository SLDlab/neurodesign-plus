# Migration To 2.0

`neurodesign-plus` 2.0 is a source-breaking release.
It keeps the public classes `Experiment`, `Design`, and `Optimisation`, but replaces the legacy event-only timing surface with a trial-aware schedule model.

## Core Model

A run contains conceptual trials.
A conceptual trial contains one or more modeled events.

For conceptual trial `j`, the realized schedule is:

1. `trial_start_j`
2. `realized_trial_start_interval_j`
3. modeled events with `realized_event_durations`
4. `realized_post_event_interval` after each event
5. `realized_event_transition_interval` only between events inside the same trial
6. `trial_end_j`
7. `realized_inter_trial_interval_j`
8. optional `realized_rest_interval_j`
9. `trial_start_(j+1)`

Only modeled event durations occupy `Xnonconv`.

## Current Public Construction Modes

### Flat One-Event Shorthand

```python
Experiment(
    ...,
    order=[0, 1, 0, 1],
    event_durations=1.0,
    trial_start_interval=0.5,
    post_event_interval=0.2,
    inter_trial_interval=2.0,
)
```

Each `order` entry is one conceptual trial containing one modeled event.

### Fixed Complete Trial Templates

```python
Experiment(
    ...,
    trial_templates=[
        {
            "template_id": "standard",
            "trial_type": "standard",
            "events": [
                {"category": "cue_easy", "code": 0, "duration": 0.8},
                {"category": "choice_left", "code": 1, "duration": 1.2},
                {"category": "feedback", "code": 2, "duration": 1.0},
            ],
        },
    ],
    trials=[{"template_id": "standard"}] * 10,
)
```

### Probabilistic Complete-Template Sampling

```python
Experiment(
    ...,
    trial_templates=[...],
    trial_template_probabilities=[0.4, 0.35, 0.25],
    n_conceptual_trials=10,
    seed=12,
)
```

Version 2.0 samples complete conceptual trials only.
It does not truncate a final template to satisfy an event budget.

## Current Timing API

Supported timing arguments:

- `event_durations`
- `trial_start_interval`
- `post_event_interval`
- `event_transition_interval`
- `inter_trial_interval`
- `rest_every_n_trials`
- `rest_interval`

Selector wrappers are supported where the public API allows them:

- `event_durations` by event category
- `trial_start_interval` by trial type
- `post_event_interval` by event category
- `event_transition_interval` by event transition

## Legacy Name Mapping

This section is intentionally historical and exists only for migration.

- `t_pre` -> `trial_start_interval`
- `t_post` -> `post_event_interval`
- `conditional_ITI` -> `event_transition_interval` and/or `inter_trial_interval`
- `ITImodel`, `ITImin`, `ITImean`, `ITImax` -> `inter_trial_interval`
- `order_keys`, `order_probabilities`, `order_length` -> `trial_templates`, `trial_template_probabilities`, `n_conceptual_trials`
- `stimuli_durations` -> `event_durations`

## Duration Grammar

Scalar shorthand remains valid:

```python
event_durations = 1.0
```

Equivalent explicit form:

```python
event_durations = {"model": "fixed", "value": 1.0}
```

Supported rule models:

- `fixed`
- `uniform`
- `exponential`
- `gaussian`

Bounded exponential and Gaussian rules use true truncated distributions.
For bounded rules, `mean` is the mean of the realized bounded distribution.

## Requested Versus Realized Timing

`Experiment.export_specification()` preserves requested public inputs.
`Design.export_payload()` preserves the realized schedule and realized timing arrays.

That separation matters because `Fe` and `Fd` depend on the realized design matrices, while `Ff` and `Fc` remain defined on the flattened realized event stream.

## Optimization Workflow

The public optimization route is:

```python
optimisation.optimise()
design = optimisation.selected_design(0)
```

Use that selected design for:

- report generation
- schedule export
- specification export
- figures
- metadata

Do not teach `.bestdesign`, direct population indexing, or selection before optimization in current user workflows.

## Deterministic RNG

Version 2.0 uses NumPy `SeedSequence` and `Generator` objects for schedule realization and optimization.
The package does not depend on global NumPy reseeding or Python `random`.

## Convergence Semantics

`convergence=k` means patience-based early stopping after `k` consecutive completed generations with no strict improvement in the generation-best objective score.

- equality counts as no improvement
- there is no minimum-delta tolerance in the current implementation
- `convergence=0` or `None` disables early stopping

## Case 10

The canonical version-2 Case 10 source lives in `validation/helpers/case10_spec.py`.
Tutorial 3, the validation comparison workflow, the determinism workflow, and manuscript-support generation all reuse that shared source.
