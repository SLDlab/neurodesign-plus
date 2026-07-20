# Tutorials

The maintained tutorial surface contains 11 authored notebooks.
They are tracked directly under `tutorials/` and executed from clean kernels during validation.

## Maintained Notebook Inventory

| Notebook | Focus |
|---|---|
| `tutorials/tutorial_1-neurodesign_base_overview.ipynb` | End-to-end overview: manual designs, crossover/mutation, and GA vs. simulation vs. random-baseline optimization |
| `tutorials/tutorial_2-comparing_designs_across_experiments.ipynb` | When raw `Fe`/`Fd` are comparable across different `Experiment` setups (same whitening matrix), and when they aren't |
| `tutorials/tutorial_3-progressive_experiment_building.ipynb` | Progressive Case 1-10 walkthrough, building from a minimal event-related design up to full branching-trial complexity |
| `tutorials/base_functions/tutorial_base-comparing_designs.ipynb` | Comparing five hand-built designs (cyclic, blocked, crossover, mutation) on `Fe`/`Fd`/`Fc`/`Ff` |
| `tutorials/base_functions/tutorial_base-designing_scoring_and_optimizing.ipynb` | Comparing GA, simulation, and random-baseline optimization via efficiency scores and simulated statistical power |
| `tutorials/base_functions/tutorial_base-discovering_best_design.ipynb` | Genetic-algorithm design discovery: comparing hand-built designs, then searching for a better one automatically |
| `tutorials/base_functions/tutorial_base-optimizing_and_reporting.ipynb` | Automatic GA optimization plus PDF report generation |
| `tutorials/new_functions/tutorial_new-event_and_trial_intervals.ipynb` | Transition-specific ITIs via `event_transition_interval` |
| `tutorials/new_functions/tutorial_new-fixed_ordering.ipynb` | Fixing the stimulus order with `order=` so optimization only searches over timing |
| `tutorials/new_functions/tutorial_new-probabilistic_ordering.ipynb` | Template-based probabilistic ordering with `order_keys` / `order_probabilities` |
| `tutorials/new_functions/tutorial_new-variable_event_durations.ipynb` | Per-stimulus variable event durations via `event_durations` |

## Workflow Rules Taught Consistently

The maintained notebooks teach these public version-2 rules consistently:

- use `Experiment` to declare the requested specification
- use `create_design(...)` or `create_manual_design(...)` for no-search workflows
- use `Optimisation` only when search is needed
- retrieve the authoritative design with `selected_design(0)` after `optimise()`
- generate reports and exports from that selected design workflow
- distinguish `n_conceptual_trials` from flattened `n_events`
- treat `Fe` and `Fd` as realized design-matrix metrics
- treat `Ff` and `Fc` as flattened event-axis metrics

Notebook execution is automated through `validation.execute_notebooks` and covered by the aggregate validation runner in `validation.run_all_validation_workflows`.
