# Tutorials

Interactive Jupyter notebook tutorials are available in the
[tutorials/ directory](https://github.com/SLDlab/neurodesign-plus/tree/master/tutorials)
of the repository.

## Overview Tutorials

| # | Tutorial | Description |
|---|---------|-------------|
| 1 | [Base Overview](https://github.com/SLDlab/neurodesign-plus/blob/master/tutorials/tutorial_1-neurodesign_base_overview.ipynb) | Introduction to the core neurodesign workflow |
| 2 | [Comparing Designs Across Experiments](https://github.com/SLDlab/neurodesign-plus/blob/master/tutorials/tutorial_2-comparing_designs_across_experiments.ipynb) | Comparing designs across diverse experiment definitions |
| 3 | [Progressive Experiment Building](https://github.com/SLDlab/neurodesign-plus/blob/master/tutorials/tutorial_3-progressive_experiment_building.ipynb) | Build event-level designs progressively from simple to complex task structures |

## Base Function Tutorials

| # | Tutorial | Description |
|---|---------|-------------|
| 4 | [Designing, Scoring, and Optimising](https://github.com/SLDlab/neurodesign-plus/blob/master/tutorials/base_functions/tutorial_base-designing_scoring_and_optimizing.ipynb) | Core workflow: designing, scoring, and optimising |
| 5 | [Comparing Designs](https://github.com/SLDlab/neurodesign-plus/blob/master/tutorials/base_functions/tutorial_base-comparing_designs.ipynb) | Side-by-side design comparison |
| 6 | [Discovering Best Design](https://github.com/SLDlab/neurodesign-plus/blob/master/tutorials/base_functions/tutorial_base-discovering_best_design.ipynb) | Finding the optimal design |
| 7 | [Optimising and Reporting](https://github.com/SLDlab/neurodesign-plus/blob/master/tutorials/base_functions/tutorial_base-optimizating_and_reporting.ipynb) | Running optimisation and generating reports |

## New Feature Tutorials

| # | Tutorial | Description |
|---|---------|-------------|
| 8 | [Controlled Probabilistic Ordering with Event Templates](https://github.com/SLDlab/neurodesign-plus/blob/master/tutorials/new_functions/tutorial_new-probabilistic_ordering.ipynb) | Sample event-level orders from probabilistic motifs rather than single-event draws |
| 9 | [Fixed Stimulus Order: Optimize Timing Without Changing the Task Sequence](https://github.com/SLDlab/neurodesign-plus/blob/master/tutorials/new_functions/tutorial_new-fixed_ordering.ipynb) | Keep the event sequence fixed while optimisation focuses on timing |
| 10 | [Transition-Specific ITIs with `conditional_ITI`](https://github.com/SLDlab/neurodesign-plus/blob/master/tutorials/new_functions/tutorial_new-conditional_ITI.ipynb) | Use transition-dependent ITI distributions keyed by `(previous_event, current_event)` |
| 11 | [Variable Stimulus Durations with `stimuli_durations`](https://github.com/SLDlab/neurodesign-plus/blob/master/tutorials/new_functions/tutorial_new-varied_stimuli_durations.ipynb) | Assign fixed or distribution-based durations per stimulus class |

:::{tip}
To run these notebooks locally, install the package in development mode and register the
Jupyter kernel. See the [installation guide](installation.md) for details.
:::
