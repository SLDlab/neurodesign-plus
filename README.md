# Neurodesign_Plus

This repository contains a extended version of the [Neurodesign](https://github.com/neuropower/neurodesign) Python package. This version extends the original capabilities to allow for more complex and specific experimental conditions, including variable stimuli durations, conditional inter-trial intervals (ITIs), and user-defined stimulus ordering.

## Neurodesign Documentation

The documentation of the base package `neurodesign` is available at [ReadTheDocs](http://neurodesign.readthedocs.io/en/latest/).

## File description

```
├── docs                    Contains the source code to generate the documentation with sphinx.
├── examples                Contains scripts to perform a design optimalisation.
└── neurodesign             Folder with the source code of the python package
    └── media               Folder contains the logo of neurodesign, which is used in the reports.
```

## Overview of Modifications

The modifications focus on the `classes.py` file, specifically enhancing the `Experiment` class to support greater flexibility. The core updates revolve around three key features:

1.  **Fixed User Input:** The ability to input a specific order and ITI sequence (bypassing randomization to evaluate a pre-determined design).
2.  **Varied Stimuli Durations:** Support for distribution-based durations rather than a single fixed duration.
3.  **Conditional ITIs:** Implementation of Inter-trial Intervals that vary based on probability distributions or the relationship between specific stimuli.

---

## Installation & Setup

See [link](./SETUP.md) for environment setup and installation instructions.

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

---

## Modifications Documentation

See [link](./TECHNICAL_CHANGES.md) for a detailed summary of changes relative to the upstream Neurodesign package.

---

## Tutorials

Tutorial notebooks / scripts will be added under `tutorials/`.
TBD.

## Credits

This is a fork of the original **Neurodesign** package.

- **Original Author:** [Neuropower Team](https://github.com/neuropower)
- **Primary refactoring, extensions, and tutorials:** Atharv Umap (Social Learning and Decisions Lab, UMD)
- **Supervision, design guidance, tutorials:** Valentin Guigon (Social Learning and Decisions Lab, UMD)
