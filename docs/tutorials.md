# Tutorials

The maintained tutorial surface contains 11 notebooks.
They are regenerated from `validation/helpers/regenerate_v2_notebooks.py` and executed from clean kernels during validation.

## Maintained Notebook Inventory

| Notebook | Focus |
|---|---|
| `tutorials/tutorial_1-neurodesign_base_overview.ipynb` | Cases 1 and 2: flat one-event shorthand and fixed conceptual trials |
| `tutorials/tutorial_2-comparing_designs_across_experiments.ipynb` | Comparison workbook for requested versus realized timing |
| `tutorials/tutorial_3-progressive_experiment_building.ipynb` | Canonical integrated Case 10 workflow |
| `tutorials/base_functions/tutorial_base-comparing_designs.ipynb` | Case 3: comparing conceptual-trial counts, event counts, and realized schedules |
| `tutorials/base_functions/tutorial_base-designing_scoring_and_optimizing.ipynb` | Case 4: scoring and patience-based stopping semantics |
| `tutorials/base_functions/tutorial_base-discovering_best_design.ipynb` | Case 5: authoritative design retrieval through `selected_design(0)` |
| `tutorials/base_functions/tutorial_base-optimizing_and_reporting.ipynb` | Case 6: report generation and reconstructable exports |
| `tutorials/new_functions/tutorial_new-event_and_trial_intervals.ipynb` | Case 7: within-trial versus between-trial interval roles |
| `tutorials/new_functions/tutorial_new-fixed_ordering.ipynb` | Case 8: fixed conceptual-trial sequences and no-optimization routes |
| `tutorials/new_functions/tutorial_new-probabilistic_ordering.ipynb` | Case 9: complete-template sampling |
| `tutorials/new_functions/tutorial_new-variable_event_durations.ipynb` | Event-duration rules and `Xnonconv` occupancy |

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

Notebook execution is automated through `validation.execute_notebooks` and covered by the aggregate validation runner in `validation.run_all`.
