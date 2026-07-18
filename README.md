# neurodesign-plus

`neurodesign-plus` is an extended and maintained fork of the original [neurodesign](https://github.com/neuropower/neurodesign) package for fMRI experimental design optimisation.

In addition to the base package workflow, `neurodesign-plus` adds support for:

1. fixed user-defined event orders via `order`,
2. variable stimulus durations via `stimuli_durations`,
3. transition-specific ITIs via `conditional_ITI`,
4. probabilistic template-based ordering via `order_keys`, `order_probabilities`, and `order_length`.

## Current version

Version 2.0 is a breaking release: experiments are now built from **trials** (each holding one or more events) instead of a flat event list, so a few options were renamed. 1.0.x scripts need these renames:

| 1.0.x                                                 | 2.0                                                                        |
| ----------------------------------------------------- | -------------------------------------------------------------------------- |
| `stimuli_durations`                                   | `event_durations`                                                          |
| `conditional_ITI`                                     | `event_transition_interval` / `inter_trial_interval`                       |
| `order_keys` / `order_probabilities` / `order_length` | `trial_templates` / `trial_template_probabilities` / `n_conceptual_trials` |

`Experiment`, `Design`, and `Optimisation` are unchanged. See the [migration
guide](MIGRATION_2.0.md) to update old code, or pin `neurodesign-plus<2.0` to
keep the old behavior.

## Installation

Install the published package with:

```bash
python -m pip install neurodesign-plus
```

## Documentation

- `neurodesign-plus` documentation: [neurodesign-plus on Read the Docs](https://neurodesign-plus.readthedocs.io/en/latest/)
- User guide: [docs/index.md](docs/index.md)
- Installation notes: [docs/installation.md](docs/installation.md)
- Migration guide (version-1 to version-2 migration details): [MIGRATION_2.0.md](MIGRATION_2.0.md)
- Technical manual: [manuals/TECHNICAL_CHANGES.md](manuals/TECHNICAL_CHANGES.md)
- Metrics guide: [manuals/METRICS.md](manuals/METRICS.md)
- Tutorials: [docs/tutorials.md](docs/tutorials.md)

## Current API Example

```python
from pathlib import Path
import json

from neurodesign import Experiment, Optimisation, report

exp = Experiment(
    TR=2.0,
    n_trials=8,
    P=[0.5, 0.5],
    C=[[1, -1]],
    rho=0.3,
    n_stimuli=2,
    event_durations=1.0,
    trial_start_interval=0.5,
    post_event_interval=0.2,
    inter_trial_interval=2.0,
    resolution=0.1,
    seed=7,
)

optimisation = Optimisation(
    experiment=exp,
    weights=[0.0, 0.5, 0.25, 0.25],
    preruncycles=1,
    cycles=1,
    optimisation="simulation",
    G=2,
    I=1,
    outdes=1,
    convergence=1,
    seed=101,
    folder=Path("output") / "readme_example",
)
optimisation.optimise()
design = optimisation.selected_design(0)

report.make_report(optimisation, Path("report.pdf"))
Path("schedule.json").write_text(
    json.dumps(design.export_payload(), indent=2), encoding="utf-8"
)
Path("specification.json").write_text(
    json.dumps(exp.export_specification(), indent=2),
    encoding="utf-8",
)
```

## Repository Layout

```text
docs/        Sphinx documentation sources
manuals/     Included long-form manuals
neurodesign/ Package source code
tests/       Automated regression and release-audit tests
tutorials/   Executed tutorial notebooks
validation/  Validation runners and reproducibility helpers
```

## Tutorials

The tutorial collection includes:

- three overview tutorials in `tutorials/`,
- four base-function tutorials in `tutorials/base_functions/`,
- four feature-focused tutorials in `tutorials/new_functions/`.

The Read the Docs tutorials page is the best place to browse the current notebook set and their intended learning progression.

## Credits

This project is a maintained fork of the original [neurodesign](https://github.com/neuropower/neurodesign) package by Joke Durnez and the Neuropower team.

Author contributions ([CRediT](https://credit.niso.org/)):

- **Atharv A. Umap** — Conceptualization, Software, Validation
- **Caroline J. Charpentier** — Project administration
- **Valentin Guigon** — Conceptualization, Methodology, Resources, Software, Supervision, Validation, Writing
