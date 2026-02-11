# Getting Started

## Overview

neurodesign-plus organises fMRI design optimisation around three core classes:

1. **`Experiment`** -- Defines the experimental specification: conditions, durations, ITI model, contrasts, and timing parameters.
2. **`Design`** -- Represents a concrete design instance: a specific stimulus order and ITI sequence generated from an Experiment.
3. **`Optimisation`** -- Searches over candidate designs using a genetic algorithm and selects designs based on efficiency metrics.

## Minimal Example

```python
import numpy as np
from neurodesign import Experiment, Optimisation

# Define the experiment
exp = Experiment(
    TR=2.0,
    n_stimuli=2,
    n_trials=20,
    P=[0.5, 0.5],
    C=np.array([[1, -1]]),
    rho=0.3,
    stim_duration=1.0,
    ITImodel="exponential",
    ITImin=1.0,
    ITImean=2.0,
    ITImax=4.0,
)

# Set up and run the optimisation
opt = Optimisation(
    experiment=exp,
    weights=[0.25, 0.25, 0.25, 0.25],
    preruncycles=2,
    cycles=5,
    seed=42,
)
opt.optimise()

# Access the best design
best = opt.bestdesign
print(f"Best overall efficiency: {best.F:.4f}")
print(f"  Fe={best.Fe:.4f}  Fd={best.Fd:.4f}  Ff={best.Ff:.4f}  Fc={best.Fc:.4f}")
```

## Metrics at a Glance

The optimiser evaluates designs on four efficiency metrics:

| Metric | Name | Measures |
|--------|------|----------|
| **Fe** | Estimation Efficiency | Precision of HRF shape estimation when events overlap |
| **Fd** | Detection Efficiency | Power to detect activation assuming canonical HRF |
| **Fc** | Confounding Efficiency | Balance of stimulus transition frequencies |
| **Ff** | Frequency Accuracy | Match between realised and target trial counts |

You control the trade-off between these metrics via the `weights` parameter. See the [metrics guide](metrics.md) for details.

## What Next?

- Learn about the [efficiency metrics](metrics.md) used to score designs
- Explore the [new features](new_features.md) added in neurodesign-plus
- See the full [API reference](api_reference.md)
- Work through the [tutorials](tutorials.md) for hands-on examples
