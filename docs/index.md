# neurodesign-plus

**Extended and maintained fork of the neurodesign package for fMRI experimental design optimisation.**

`neurodesign-plus` extends the original [neurodesign](https://github.com/neuropower/neurodesign) package with support for variable stimulus durations, conditional inter-trial intervals, user-defined stimulus ordering, and probabilistic ordering.

## Quick Install

```bash
pip install neurodesign-plus
```

## Key Features

- **Genetic algorithm optimisation** of fMRI experimental designs
- **Variable stimulus durations** -- specify per-condition distribution-based durations
- **Conditional ITIs** -- inter-trial intervals that depend on stimulus transitions
- **Fixed or probabilistic ordering** -- inject user-defined or probability-sampled stimulus orders
- **Four efficiency metrics** -- Estimation (Fe), Detection (Fd), Confounding (Fc), Frequency (Ff)
- **PDF reports** -- automatic generation of optimisation result summaries

```{toctree}
:maxdepth: 2
:caption: User Guide

installation
getting_started
metrics
new_features
```

```{toctree}
:maxdepth: 2
:caption: Reference

api_reference
tutorials
changelog
```
