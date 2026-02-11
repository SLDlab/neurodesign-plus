# Neurodesign_Plus

This repository contains a extended version of the [Neurodesign](https://github.com/neuropower/neurodesign) Python package. This version extends the original capabilities to allow for more complex and specific experimental conditions, including variable stimuli durations, conditional inter-trial intervals (ITIs), and user-defined stimulus ordering.

## Neurodesign Documentation

The documentation of the base package `neurodesign` is available at [ReadTheDocs](http://neurodesign.readthedocs.io/en/latest/).

## File description

```
├── docs                    Contains the source code to generate the documentation with sphinx (WIP).
├── manuals                 Contains comprehensive markdown documentation on neurodesign-plus.
└── neurodesign             Folder with the source code of the python package
    └── media               Folder contains the logo of neurodesign, which is used in the reports.
├── tests                   Contains scripts to test modifications to the package.
└── tutorials               Contains .ipynb tutorials on neurodesign-plus, the base functions and the new functions.
```

## Overview of Modifications

The modifications focus on the `classes.py` file, specifically enhancing the `Experiment` class to support greater flexibility. The core updates revolve around three key features:

1.  **Fixed User Input:** The ability to input a specific order and ITI sequence (bypassing randomization to evaluate a pre-determined design).
2.  **Varied Stimuli Durations:** Support for distribution-based durations rather than a single fixed duration.
3.  **Conditional ITIs:** Implementation of Inter-trial Intervals that vary based on probability distributions or the relationship between specific stimuli.

---

## Installation & Setup

See [link](./manuals/SETUP.md) for environment setup and installation instructions.

---

## Original Tool

The Neurodesign package is organized around three core classes: Experiment, Design, and Optimization.

1. Experiment defines the experiment specification (conditions/stimuli, durations, ITIs, constraints).

2. Design represents a concrete design instance generated from an experiment (event order + ITIs).

3. Optimization searches over candidate designs and selects designs based on efficiency metrics.

## Metrics

We are looking to optimize experimental design, but what defines good metrics?

The Neurodesign Python package optimizes these experiments based on four metrics:

1. **Fe : Estimation Efficiency**

   Measures how well the design allows estimation of response shape and amplitude when events overlap in time.
   High Fe means the model can separate the contributions of different events cleanly.

2. **Fd : Detection Efficiency**

   Measures how well the design supports detecting differences between conditions assuming a fixed canonical HRF.
   High Fd means convolved regressors for each condition are distinguishable.

3. **Fc: Confounding Efficiency**

   Measures how closely the realized number of trials per condition matches the prescribed probabilities (e.g., 50/50).
   High Fc means the realized proportions match what was specified.

4. **Ff : Frequency Accuracy**

   Measures whether the condition sequence is balanced across time and transitions (immediate and longer-range).
   High Ff means the design reduces serial dependencies and avoids systematic transition biases.

During optimization, you can assign weights to these metrics to reflect what matters most for your experiment.

**See [link](./manuals/METRICS.md) for guidelines on interpreting metrics when optimising designs.**

---

## Modifications Documentation

See [link](./manuals/TECHNICAL_CHANGES.md) for a detailed summary of changes relative to the upstream Neurodesign package.

With these new parameters and changes in the package, it is simply a matter of defining the parameters required for the specific use case and the tool will perform the optimization.

**Note** that precedence for order follows as (if all provided):

- Fixed ordering
- Controlled ordering
- Random ordering

The same holds for conditional_ITI and stimuli_durations.

---

## Tutorials

Tutorial notebooks / scripts are under `tutorials/`.

```
tutorials
├── base_functions                                    Contains tutorials of the different base package functions.
├── new_functions                                     Contains tutorials on using the modifications.
├── tutorial_1_neurodesign_base_overview              Base tutorial of the neurodesign package.
└── tutorial_2_comparing_designs_across_experiments   Tutorial on comparing designs across diverse experiment definitions.
```

For quick access, find the tutorials in the table below:
| # | Tutorial | Link |
| :---: | :--- | :---: |
| 1 | **tutorial_1_neurodesign_base_overview** | [View](./tutorials/tutorial_1_neurodesign_base_overview.ipynb) |
| 2 | **tutorial_2_comparing_designs_across_experiments** | [View](./tutorials/tutorial_2_comparing_designs_across_experiments.ipynb) |
| 3 | **tutorial_base_compare_and_simulate** | [View](./tutorials/base_functions/tutorial_base_compare_and_simulate.ipynb) |
| 4 | **tutorial_base_comparing_designs** | [View](./tutorials/base_functions/tutorial_base_comparing_designs.ipynb) |
| 5 | **tutorial_base_discovering_best_design** | [View](./tutorials/base_functions/tutorial_base_discovering_best_design.ipynb) |
| 6 | **tutorial_base_optimizating_and_report** | [View](./tutorials/base_functions/tutorial_base_optimizating_and_report.ipynb) |
| 7 | **tutorial_new_controlled_probabalistic_ordering** | [View](./tutorials/new_functions/tutorial_new_controlled_probabalistic_ordering.ipynb) |
| 8 | **tutorial_new_fixed_order** | [View](./tutorials/new_functions/tutorial_new_fixed_order.ipynb) |
| 9 | **tutorial_new_varied_ITI** | [View](./tutorials/new_functions/tutorial_new_varied_ITI.ipynb) |
| 10 | **tutorial_new_varied_stimuli_durations** | [View](./tutorials/new_functions/tutorial_new_varied_stimuli_durations.ipynb) |

## Credits

This is a fork of the original **Neurodesign** package.

- **Original Author:** [Neuropower Team](https://github.com/neuropower)
- **Primary refactoring, extensions, and tutorials:** Atharv Umap (Social Learning and Decisions Lab, UMD)
- **Supervision, design guidance, tutorials:** Valentin Guigon (Social Learning and Decisions Lab, UMD)
