# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
### Added

### Changed

### Deprecated

### Removed

### Fixed

### Security
-->

## [2.0.0] - 2026-07-18

### Breaking changes

- Replaced the inherited event-only timing surface with a trial-aware version-2 architecture based on conceptual trials and modeled events.
- Replaced ambiguous ITI handling with separate `trial_start_interval`, `post_event_interval`, `event_transition_interval`, `inter_trial_interval`, `rest_every_n_trials`, and `rest_interval`.
- Replaced event-level probability-template truncation with complete-template conceptual-trial sampling driven by `trial_templates`, `trial_template_probabilities`, and `n_conceptual_trials`.
- Replaced direct manual `Design(order=..., ITI=...)` construction with `Experiment.create_design(...)` and `Experiment.create_manual_design(...)`.
- Standardized the public optimization workflow around `optimise()` followed by `selected_design(0)`.

### Added

- Flat one-event shorthand through `order`.
- Fixed complete conceptual-trial templates through `trial_templates` plus `trials`.
- Probabilistic complete-template sampling through `trial_templates`, `trial_template_probabilities`, and `n_conceptual_trials`.
- Shared timing-rule parsing for `event_durations`, `trial_start_interval`, `post_event_interval`, `event_transition_interval`, `inter_trial_interval`, and `rest_interval`.
- Selector wrappers for trial-type, event-category, and event-transition timing rules.
- Requested, normalized, and realized timing state exports.
- Trial-aware schedule metadata on `Design`, including trial IDs, template IDs, trial types, and realized schedule arrays.
- Public `selected_design(rank)` retrieval after optimization.
- Canonical migration guide in `MIGRATION_2.0.md`.

### Fixed

- `Xnonconv` occupancy now reflects modeled event durations only rather than absorbing surrounding intervals.
- Trial-start, post-event, within-trial transition, between-trial, and rest timing roles now remain structurally distinct.
- Rest intervals now remain boundary intervals instead of being treated like synthetic task events.
- `Ff` and `Fc` now use flattened realized event counts rather than conceptual-trial counts.
- Case 10 now uses one canonical version-2 configuration shared across tutorials, validation, and manuscript-support generation.

### Validation and reproducibility

- Switched schedule sampling and optimization RNG handling to NumPy `SeedSequence` and `Generator`.
- Added deterministic canonical Case 10 validation, manuscript-support generation, and cross-process determinism checks.
- Added release-audit coverage for `selected_design()` guardrails, bounded distribution semantics, and version-2 public workflows.
- Regenerated and validated the maintained tutorial notebooks from a single canonical notebook generator.

### Migration requirements

- Version 2.0 is intentionally source-breaking for scripts written against the version-1 timing API.
- User-facing docs now route version migration through `MIGRATION_2.0.md`.
- Tutorials and examples now teach the selected-design workflow, trial-aware timing terms, and event-count versus conceptual-trial-count distinctions.

### Removed arguments

- Removed `stimuli_durations`.
- Removed `conditional_ITI`.
- Removed `order_keys`, `order_probabilities`, and `order_length`.
- Removed `all_stim_durations`.
- Removed `t_pre` and `t_post`.
- Removed `ITImodel`, `ITImin`, `ITImean`, and `ITImax`.
- Removed active user-facing reliance on `.bestdesign` and internal design-pool indexing.

### Export and report changes

- Reports now align with the selected-design workflow.
- `Design.export_payload()` now preserves reconstructable schedule arrays, schedule rows, counts, and metric components.
- `Experiment.export_specification()` now preserves requested public timing specifications and trial-template configuration.

### Trial-aware optimization changes

- Flat mode remains event-order based.
- Fixed conceptual-trial mode preserves the declared trial sequence.
- Probabilistic template mode mutates and crosses at conceptual-trial boundaries instead of truncating event-level template fragments.

### RNG changes

- Removed dependence on global NumPy reseeding and Python `random`.
- RNG derivation is now stable across sampling, crossover, mutation, and immigration.

### Convergence changes

- `convergence=k` now documents and exposes patience-based early stopping after `k` completed generations without strict improvement in the generation-best objective score.
- `convergence=0` and `convergence=None` disable early stopping.
- Public optimization state now exposes `generations_completed`, `stop_reason`, and `optima` for user-facing reporting.

## [1.0.2] - 2026-03-25

### Added

- Added `tutorial_3_progressive_experiment_building.ipynb` to document progressive experiment construction in `neurodesign-plus`.

### Changed

- Rewrote the tutorials in `tutorials/new_functions/` to align their communication style with the main tutorial set, with clearer conceptual framing, parameter explanations, and result interpretation.
- Modified tutorial names for better usecases identification.
- Updated the Read the Docs tutorial index to reflect the renamed notebooks and to include `tutorial_3_progressive_experiment_building.ipynb`.
- Synchronized the technical documentation with the current codebase behavior, including the documented semantics of `order_length`, `sample_from_probabilities()`, and `max_eff()`.
- Updated `README.md` and notebook test configuration to match the current tutorial names and documentation structure.

## [1.0.1] - 2026-02-16

### Added

- Added authors contacts to `README.md` and `pyproject.toml`.

## [1.0.0] - 2026-02-11

First release of `neurodesign-plus` on PyPI.

### Added

- Variable stimulus durations via `stimuli_durations` parameter on `Experiment`.
- Conditional ITI distributions via `conditional_ITI` parameter on `Experiment`.
- Fixed user-defined stimulus ordering via `order` parameter on `Experiment`.
- Probabilistic ordering via `order_probabilities`, `order_keys`, `order_length` parameters.
- New static methods: `sample_stim_durations`, `generate_iti`, `calculate_duration`, `sample_from_probabilities`.
- Automated testing with tox, pytest, and GitHub Actions.
- Comprehensive documentation hosted on ReadTheDocs.
- 10 tutorial notebooks covering base and new features.
- Detailed manuals on metrics, installation, and technical changes.

### Changed

- Published on PyPI as `neurodesign-plus` (import remains `import neurodesign`).
- Classes Experiment, Design and Optimisation have been renamed (capitalised).
- Experiment-Design separation enforced: per-trial arrays stored on Design, specifications on Experiment.
- Optimisation skips Fe/Fd calibration pre-runs when corresponding weight is zero.
- Code formatted with black.
- flake8 errors were fixed.
- Packaging reorganized to use hatch and pyproject.toml.
- Use rich for progress bar.

## [0.2.02] - 2018-08-24

- bug with timepoints fix

## [0.2.01] - 2018-02-09

- bug in report.py

## [0.2.00] - 2018-02-09

- change names to reflect new options
- update examples
- reflects JSS submission

## [0.1.13] - 2018-02-06

- fix on not doing FdMax and FeMax if not necessary

## [0.1.12] - 2018-02-06

- fix on matplotlib bug

## [0.1.08]-11 - 2018-02-06

- fix on duration bug

## [0.1.07] - 2018-02-05

- added option for random design generation

## [0.1.06] - 2018-11-22

- debug conda matplotlib
- remove shift in onsets
- ensure ITI is nominal value
- improve handling of resolution
- improve handling of hrf (resolution)

## [0.1.05] - 2017-11-22

- report: remove bug ndarray check in report

## [0.1.04] - 2017-08-30

- add hrf_precision as parameter: not same as resolution

## [0.1.03] - 2017-08-09

- remove indentation bugs
- replace often reoccuring warnings with internal warning

## [0.1.02] - 2017-07-31

- debug first ITI (should be 0)
- debug Hardprob
- debug maximum ITI with resolution
- round ITI to resolution before computing onsets
- added exceptions for longer designs

## [0.1.01] - 2017-07-03

- debug numpy exception

## [0.1.00] - 2017-04-26

- package underwent extensive testing through the online application
- examples are added complementary to the manuscript under revision

## [0.0.28] - 2017-02-20

- debug correlation check

## [0.0.27] - 2017-02-20

- debug rest durations
- debug repeated msequences

## [0.0.26] - 2017-02-14

- patch reoccurring (but rare) problem with too long a design

## [0.0.25] - 2017-02-06

- avoid getting stuck while trying to find blocked designs

## [0.0.24] - 2017-02-02

- add some extra space for max duration (0.5s) to avoid trials exceeding the experiment duration

## [0.0.23] - 2017-02-01

- remove requirements for now

## [0.0.22] - 2017-02-01

- update requirements

## [0.0.21] - 2017-02-01

- remove itertools from requirements

## [0.0.20] - 2017-02-01

- remove collections from requirements

## [0.0.19] - 2017-02-01

- pypi problems

## [0.0.18] - 2017-02-01

- remove typo in install_requires

## [0.0.17] - 2017-02-01

- remove requirements.txt

## [0.0.16] - 2017-02-01

- debug memory problem induced in 0.0.15

## [0.0.15] - 2017-01-31

- debug LinAlgError
- debug necessity for calculating both Fe and Fd --> speed up
- debug initial population weights

## [0.0.14] - 2017-01-19

- improve which designs are outputted
- improve report

## [0.0.13] - 2017-01-18

- debug max number of repeats

## [0.0.12] - 2017-01-18

- debug generation of ITI's from truncated distribution

## [0.0.11] - 2017-01-17

- debug generation of ITI's from truncated distribution

## [0.0.10] - 2017-01-10

- remove direct support for GUI

## [0.0.9] - 2016-12-19

- change matplotlib backend for report

## [0.0.8] - 2016-12-19

- debug script

## [0.0.7] - 2016-12-09

- debug seeds
- restructure a few parameters in other classes

## [0.0.6] - 2016-11-18

- add report to download

## [0.0.2] - 2016-11-18

- remove bug in report when numpy array is given.
