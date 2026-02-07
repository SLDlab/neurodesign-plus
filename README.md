# Neurodesign_Plus

This repository contains a modified version of the [Neurodesign](https://github.com/neuropower/neurodesign) Python package. This refactor extends the original capabilities to allow for more complex and specific experimental conditions, including variable stimuli durations, conditional inter-trial intervals (ITIs), and user-defined stimulus ordering.

## Neurodesign Documentation

The documentation of neurodesign is available at [ReadTheDocs](http://neurodesign.readthedocs.io/en/latest/).

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

Here is the [link](./SETUP.md) to the guide of setting up the tool and its require environment 

---
## Original Tool and Metrics
The Neurodesign package has three different classes: Design, Experiment, and Optimization.

1. **Experiment** class defines an experiment with the stimuli, their duration, ITI duration, etc.

2. **Design** class defines one possible design for an experiment based on a given order and ITI

3. **Optimization** runs through multiple designs and finds the best design based on the metrics


Now given these classes and this tool, we are looking to optimize experimental design, but what defines good metrics?

The Neurodesign Python package optimizes these experiments based on four metrics: 

1. **Fe : Estimation Efficiency**
    
    Measures how well the design allows to estimate shape and amplitude of neural responses when events overlap in time. 
    High Fe: the model can separate the contributions of different events cleanly.

2. **Fd : Detection Efficiency**

    Measures how well the design supports detecting differences between conditions assuming a fixed canonical HRF. 
    High Fd: convolved regressors for each condition are distinguishable.

3. **Fc: Confounding Efficiency**

    Measures how closely the design’s actual number of trials per condition matches the desired probabilities (e.g., 50/50). 
    High Ff means the realized proportions of conditions are very close to what has been prescribed to the toolbox. 

4. **Ff : Frequency Accuracy**

    Measures whether the sequence of conditions is well balanced across time, looking at immediate and distant transitions. High Fc means the design avoids serial dependencies and makes every condition equally likely to follow any other.   

In your experiment, you will be able to optimize based on these four metrics giving a weight marking the importance of each of these metrics.

---
## Modifications Documentation

Here is the [link](./TECHNICAL_CHANGES.md) to the modification guideline to the classes.py file from the original Neurodesign package.
___

## Tutorials

Follow the following files for a tutorial on the following features...

TBD
___


## Credits

This is a fork of the original **Neurodesign** package.
* **Original Author:** [Neuropower Team](https://github.com/neuropower)
* **Modifications by:** Atharv Umap, Valentin Guigon ([Social Learning and Decisions Lab](https://sldlab.umd.edu/))