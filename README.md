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

## Neurodesign-plus Documentation

The complete documentation for `neurodesign-plus` is hosted on [ReadTheDocs](https://neurodesign-plus.readthedocs.io/en/latest/). It serves as your primary resource for:

1.  **Getting Started:** Installation and setup.
2.  **Core Concepts:** Understanding Neurodesign Efficiency Metrics.
3.  **Advanced Usage:** Implementing fixed user inputs, variable durations, and conditional ITIs.
4.  **Tutorials:** Jupyter notebook tutorials to fast-track your learning.

---

## Credits

This is a fork of the original **Neurodesign** package.

- **Original Author:** [Neuropower Team](https://github.com/neuropower)
- **Primary refactoring, extensions, and tutorials:** Atharv Umap (Social Learning and Decisions Lab, UMD)
- **Supervision, design guidance, tutorials:** Valentin Guigon (Social Learning and Decisions Lab, UMD)
