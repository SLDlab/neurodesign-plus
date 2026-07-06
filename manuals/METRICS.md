# Neurodesign-Plus 2.0 Metrics Guide

## Scope

Version 2.0 keeps the four public design metrics:

- `Fe`
- `Fd`
- `Ff`
- `Fc`

The scheduling model is now trial-aware, but the metric axes remain event-aware where appropriate:

- `Fe` and `Fd` depend on the realized timing arrays and the realized design matrices
- `Ff` and `Fc` are computed over flattened event categories
- `Ff` and `Fc` therefore use `n_events`, not `n_conceptual_trials`

## Conceptual Trials Versus Flattened Events

Conceptual trials determine:

- trial-template sampling
- trial boundaries
- `trial_start_interval`
- `inter_trial_interval`
- `rest_interval`

Flattened modeled events determine:

- event rows in the exported schedule
- event occupancy in `Xnonconv`
- `Ff`
- `Fc`

In a flat one-event design, `n_conceptual_trials == n_events`.
In a fixed or probabilistic template design, `n_events` is usually larger because each conceptual trial can contain multiple events.

## Design-Matrix Dependence

`Fe` is computed from the deconvolved design matrix.
`Fd` is computed from the convolved design matrix.

Both metrics depend on the realized timing state of the sampled `Design`, including:

- realized event durations
- realized within-trial transition intervals
- realized between-trial intervals
- realized rest boundaries

That means two designs drawn from the same `Experiment` can produce different `Fe` and `Fd` values when their realized timing differs.

`Xnonconv` contains occupancy only during modeled event durations.
It is zero during:

- `trial_start_interval`
- `post_event_interval`
- `event_transition_interval`
- `inter_trial_interval`
- `rest_interval`

`Xconv` can remain nonzero during those periods because the HRF persists after modeled event offset.

## Event-Axis Metrics

`Ff` measures how closely the realized flattened event counts match the target category probabilities `P`.

`Fc` measures balance of flattened event-category transitions up to the configured confound order.

For fixed and probabilistic template designs:

- `Ff` does not become a trial-template frequency metric
- `Fc` does not become a trial-template transition metric
- both remain defined on the flattened event stream produced by the realized schedule

## Raw Versus Normalized Values

`Ff` and `Fc` are normalized against event-count-dependent reference mismatches, so they remain on the familiar balance scale used by the package.

`Fe` and `Fd` are divided by `Experiment.FeMax` and `Experiment.FdMax`.
If those maxima remain at their default value of `1`, `Optimisation.optimise()` estimates empirical prerun references when the corresponding weights are positive.
Because those references are empirical rather than mathematical upper bounds, a later selected design can exceed `1.0`.

Interpretation rule:

- within one optimization run, larger values are better
- values above `1.0` for `Fe` or `Fd` mean the selected design beat the empirical prerun reference, not that the implementation is wrong

## Weighted Objective

`Optimisation` combines the component scores as:

```python
F = w_fe * Fe + w_fd * Fd + w_ff * Ff + w_fc * Fc
```

The objective uses the scores stored on the realized `Design`.

## Current Flat Example

This example is covered by the release-audit tests.

```python
from neurodesign import Experiment

exp = Experiment(
    TR=2.0,
    n_trials=8,
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

design = exp.create_design(seed=7)
design.designmatrix().FCalc(weights=[0.0, 0.5, 0.25, 0.25])
```

## Current Optimisation Example

This workflow is also exercised by the release-audit tests.

```python
from neurodesign import Experiment, Optimisation

exp = Experiment(
    TR=2.0,
    n_trials=8,
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

optimisation = Optimisation(
    experiment=exp,
    weights=[0.0, 0.5, 0.25, 0.25],
    preruncycles=1,
    cycles=1,
    optimisation="simulation",
    G=2,
    I=1,
    outdes=1,
    convergence=1,
    seed=101,
)
optimisation.optimise()
design = optimisation.selected_design(0)
```

Useful inspection points:

- `design.export_payload()["counts"]["n_events"]`
- `optimisation.exp.export_specification()["n_conceptual_trials"]`
- `optimisation.generations_completed`
- `optimisation.stop_reason`
- `design.Fe`, `design.Fd`, `design.Ff`, `design.Fc`, `design.F`

## Convergence

`convergence=k` means patience-based early stopping after `k` consecutive completed generations with no strict improvement in the generation-best objective score.

- equality counts as no improvement
- there is no minimum-delta tolerance in the current implementation
- early stopping does not prove a global optimum

When optimization is disabled entirely, metric calculations on a direct `Design` still use the same definitions.
