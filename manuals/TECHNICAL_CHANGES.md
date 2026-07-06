# Technical Changes In 2.0

## Authoritative Implementation

Version 2.0 keeps the public classes:

- `Experiment`
- `Design`
- `Optimisation`

The authoritative implementation lives in [neurodesign/classes.py](../neurodesign/classes.py).
Both public import paths resolve to those same class objects:

```python
from neurodesign import Experiment, Design, Optimisation
from neurodesign.classes import Experiment, Design, Optimisation
```

## Trial-Aware Normalization

`Experiment` now normalizes three construction modes:

- flat one-event shorthand
- fixed complete trial-template sequences
- probabilistic complete-template sampling

For template modes, the implementation separates:

- template definitions
- realized trial instances
- flattened modeled events

Normalization establishes:

- `n_conceptual_trials`
- template IDs
- trial types
- event categories and event codes
- event-, trial-, within-trial-transition-, and between-trial cardinalities

Version 2.0 does not truncate templates to fit an event budget.
Probabilistic template mode samples whole conceptual trials only.

## Timing-Rule Parsing

All public timing components share one parser:

- `event_durations`
- `trial_start_interval`
- `post_event_interval`
- `event_transition_interval`
- `inter_trial_interval`
- `rest_interval`

Supported rule forms are:

- scalar shorthand
- explicit `{"model": ...}` rules
- selector wrappers such as `by_event_category`, `by_trial_type`, and `by_event_transition`

Normalization converts those public forms into canonical internal rule objects before any schedule is realized.

## Requested, Normalized, And Realized State

Version 2.0 keeps three separate timing layers:

1. requested public specifications on `Experiment`
2. normalized internal rules on `Experiment`
3. realized occurrence-level values on `Design`

`Experiment.export_specification()` preserves the requested surface.
`Design.export_payload()` preserves the realized schedule, realized timing arrays, metrics, and counts.

## Bounded Distribution Semantics

Uniform, exponential, and Gaussian timing rules now use one common validation and sampling path.

Important behavior changes:

- bounded exponential rules use a true truncated exponential
- bounded Gaussian rules use a true truncated normal
- bounded means are interpreted as the mean of the realized bounded distribution
- values are rounded once to the experiment resolution

The implementation no longer depends on draw-then-clip behavior.

## RNG Architecture

Version 2.0 uses deterministic NumPy `SeedSequence` and `Generator` objects throughout:

- design realization
- template sampling
- duration sampling
- interval sampling
- mutation
- crossover
- immigration

The package does not rely on Python `random.choices` or global NumPy reseeding.

## Schedule Representation

`Design` now stores a realized trial-aware schedule with explicit metadata, including:

- flattened event `order`
- event categories
- realized event durations
- trial IDs
- event index within each trial
- trial template IDs
- trial type IDs
- realized trial-start intervals
- realized post-event intervals
- realized within-trial transition intervals
- realized between-trial intervals
- realized rest intervals
- event onsets and offsets
- trial starts and ends

Rests remain boundary intervals.
They are not modeled as synthetic events.

## Trial-Boundary-Aware Optimization

`Optimisation` now respects the schedule construction mode:

- flat mode mutates and crosses event orders
- fixed template mode preserves the fixed conceptual-trial sequence
- probabilistic template mode mutates and crosses at conceptual-trial boundaries

After any sequence change, the package resamples the timing state needed for the new sequence, rebuilds the schedule, and recalculates the design matrices and objective metrics.

## Public Selection Workflow

The authoritative public route is:

```python
optimisation.optimise()
design = optimisation.selected_design(0)
```

`selected_design(rank)` retrieves the clustered output representatives created by `evaluate()`.
Calling `selected_design()` before `optimise()` raises a runtime error.
Out-of-range ranks raise `IndexError`.

User-facing workflows should not rely on:

- `.bestdesign`
- direct population indexing
- re-sorting internal design pools

## Patience-Based Early Stopping

`Optimisation(convergence=k)` implements patience-based early stopping.

- `k > 0`: stop after `k` consecutive completed generations without strict improvement in the generation-best objective score
- `k in {0, None}`: disable early stopping
- equality counts as no improvement
- there is no minimum-delta tolerance in the current implementation

The public optimization state records:

- `generations_completed`
- `stop_reason`
- `optima`
- `bestdesign_generation`

## Reports And Export Payloads

Version 2.0 reporting and export surfaces now align around the selected design workflow.

Reports are generated from the `Optimisation` instance.
Exports preserve enough information to reconstruct the selected schedule:

- schedule table rows
- realized schedule arrays
- trial and template metadata
- counts
- metric components
- requested experiment specification

## Validation Runner

The maintained validation surface is organized through:

- `validation/manifest.py`
- `validation/run_all.py`
- `validation/execute_notebooks.py`
- `validation/helpers/regenerate_v2_notebooks.py`

That runner covers:

- the full pytest suite
- notebook execution from clean kernels
- offline docs builds with warnings as errors
- canonical Case 10 comparison
- manuscript-support Case 10 generation
- determinism checks
- timing-architecture asset generation
