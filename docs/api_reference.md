# API Reference

## Core Classes

The package exposes three primary classes through `neurodesign`:

```python
from neurodesign import Experiment, Design, Optimisation
```

The compatibility import path `from neurodesign.classes import ...` resolves to the same class objects implemented in `neurodesign/classes.py`.

Public construction and selection routes:

- create sampled designs with `Experiment.create_design(...)`
- create fully manual flat one-event designs with `Experiment.create_manual_design(...)`
- run search with `Optimisation.optimise()`
- retrieve reported designs with `Optimisation.selected_design(rank)`

Direct `Design(...)` construction is not a user-facing version-2 workflow.

### Experiment

```{eval-rst}
.. autoclass:: neurodesign.Experiment
   :members:
   :show-inheritance:
```

### Design

```{eval-rst}
.. autoclass:: neurodesign.Design
   :members:
   :show-inheritance:
```

### Optimisation

```{eval-rst}
.. autoclass:: neurodesign.Optimisation
   :members:
   :show-inheritance:
```

## Utility Modules

### generate

```{eval-rst}
.. automodule:: neurodesign.generate
   :members:
```

### msequence

```{eval-rst}
.. automodule:: neurodesign.msequence
   :members:
```

### report

```{eval-rst}
.. automodule:: neurodesign.report
   :members:
```
