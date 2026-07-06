# neurodesign-plus

`neurodesign-plus` is the distribution name for the version-2.0.0 trial-aware release.
The Python import namespace remains `neurodesign`.

Version 2.0 models a run as conceptual trials containing one or more modeled events.
That public model supports:

- flat one-event shorthand through `order`
- fixed complete trial templates through `trial_templates` plus `trials`
- probabilistic complete-template sampling through `trial_templates`, `trial_template_probabilities`, and `n_conceptual_trials`
- explicit timing roles through `event_durations`, `trial_start_interval`, `post_event_interval`, `event_transition_interval`, `inter_trial_interval`, and `rest_interval`
- separate requested specifications, normalized rules, and realized timing arrays on the sampled `Design`

Optimization stays public through `Optimisation`. After `optimise()`, retrieve the authoritative reported design with `selected_design(0)`, then drive reporting and exports from that selected design. Reports are generated with `neurodesign.report`, and both schedules and experiment specifications can be exported in reconstructable JSON form. Seeded workflows use deterministic NumPy `SeedSequence` and `Generator` plumbing throughout design sampling and optimization.

Install the published package with:

```bash
python -m pip install neurodesign-plus
```

For source, development, notebook, and documentation installs, see [manuals/SETUP.md](manuals/SETUP.md) and [docs/installation.md](docs/installation.md). For version-1 to version-2 migration details, see [MIGRATION_2.0.md](MIGRATION_2.0.md).

## Current API Example

This public workflow is covered by the release-audit tests and notebook generator.

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

## Documentation

- User guide: [docs/index.md](docs/index.md)
- Installation notes: [docs/installation.md](docs/installation.md)
- Migration guide: [MIGRATION_2.0.md](MIGRATION_2.0.md)
- Technical manual: [manuals/TECHNICAL_CHANGES.md](manuals/TECHNICAL_CHANGES.md)
- Metrics guide: [manuals/METRICS.md](manuals/METRICS.md)
- Tutorials: [docs/tutorials.md](docs/tutorials.md)

## Repository Layout

```text
docs/        Sphinx documentation sources
manuals/     Included long-form manuals
neurodesign/ Package source code
tests/       Automated regression and release-audit tests
tutorials/   Executed tutorial notebooks
validation/  Validation runners and reproducibility helpers
```

## Credits

This project is a maintained fork of the original [neurodesign](https://github.com/neuropower/neurodesign) package.

- Original author: Joke Durnez and the Neuropower team
- neurodesign-plus maintenance and tutorials: Atharv Amar Umap
- supervision and design guidance: Valentin Guigon
