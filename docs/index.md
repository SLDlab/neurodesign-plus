# neurodesign-plus 2.0

**Trial-aware fMRI experimental design optimisation for flat one-event and multi-event conceptual trials.**

`neurodesign-plus` is the install distribution.
`neurodesign` is the Python import package.

Version 2.0 keeps the public classes `Experiment`, `Design`, and `Optimisation`, but the active teaching surface is now trial-aware:

- conceptual trials can contain one or more modeled events
- flat `order` inputs are shorthand for one-event conceptual trials
- fixed workflows use complete trial templates
- probabilistic workflows sample complete templates for `n_conceptual_trials`
- timing is separated into event, within-trial, between-trial, and optional rest roles
- requested specifications, normalized rules, and realized timing arrays remain distinct

Version 2.0 is intentionally source-breaking.
For legacy-to-current mapping, use the [migration guide](migration.md).

```{toctree}
:maxdepth: 2
:caption: User Guide

installation
getting_started
migration
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
