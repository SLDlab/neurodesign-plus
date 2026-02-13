# Technical Documentation: Refactoring Details

This document outlines the specific internal modifications made to the `classes.py` file in the Neurodesign package. It details new class variables, added functions, and modifications to existing logic within the `Design`, `Experiment`, and `Optimisation` classes.

---

## Architecture Overview

The refactored code enforces the strict separation between the **Experiment** (a fixed container defining the experimental context) and the **Design** (a specific trial sequence within that container).

- **Experiment** stores distribution *specifications* (ITI model parameters, stimulus duration specs) and computes the whitening matrix, timepoints, and HRF components once. It does not hold per-trial timing arrays.
- **Design** stores concrete per-trial arrays (order, ITI, stimulus durations) sampled from the Experiment's specifications. All designs in an optimization population share a single Experiment object.

This guarantees that all efficiency metrics (Fe, Fd) are computed against the same whitening matrix, making them comparable across designs.

---

## 1. Added Class Variables

### **Design Class**

* **`all_stim_durations`** `(list of floats or None)`
    * Concrete per-trial stimulus durations for this design, including `t_pre` and `t_post`.
    * `None` when all stimuli share the same `stim_duration` (original package behavior).
    * When set, `designmatrix()` uses these per-trial values instead of `experiment.stim_duration`.
    * Sampled via `Experiment.sample_stim_durations()` and passed at construction: `Design(order=..., ITI=..., experiment=..., all_stim_durations=...)`.

### **Experiment Class**

* **`stimuli_durations`** `(list of int/float or dicts, or None)`
    * A specification template — one entry per stimulus — defining how to sample each stimulus's duration.
    * Each entry is either a scalar (fixed duration) or a dict specifying a distribution:
        ```python
        stimuli_durations = [
            {"model": "fixed", "mean": 1.0},
            {"model": "exponential", "mean": 2.0, "min": 1.0, "max": 5.0},
            1.5,  # scalar shorthand for fixed
        ]
        ```
    * Length must equal `n_stimuli`. Requires `trial_max` to be specified.
    * If not provided, all stimuli use the original `stim_duration`.

* **`conditional_ITI`** `(dict or None)`
    * Specification for condition-dependent ITI distributions. Keys are `(prev_stim, curr_stim)` tuples or `"default"`. Values are dicts with `"model"`, `"mean"`, and optional `"min"`, `"max"`, `"std"`.
    * **Example:**
        ```python
        conditional_ITI = {
            (0, 1): {"model": "exponential", "mean": 2, "min": 1},
            (1, 2): {"model": "fixed", "mean": 4},
            "default": {"model": "exponential", "mean": 3, "min": 1},
        }
        ```

* **`order`** `(list of ints or None)`
    * A user-provided fixed stimulus order. When provided, `order_fixed` is set to `True`.
    * The order is preserved across all designs during optimization — crossover and mutation do not modify it.

* **`trial_max`** `(float)`
    * The maximum stimulus duration across all conditions. Required when `stimuli_durations` is provided. Defaults to `stim_duration` otherwise.
    * Used to compute the container's `trial_duration = trial_max + t_pre + t_post`, which determines the experiment's total duration and whitening matrix dimensions.

#### **Sequence Generation Variables**

* **`order_keys`** `(list of integer lists)` — Sequences of stimuli used as sampling units.
* **`order_probabilities`** `(list of floats)` — Probabilities for each key (must sum to 1).
* **`order_length`** `(int)` — Number of draws from the key distribution.

These three are inputs for `sample_from_probabilities()`. When `order_probabilities` is provided, the optimizer generates stimulus orders by sampling from these sequence distributions rather than using the original blocked/random/m-sequence generators.

---

## 2. New Functions

### **Experiment Class (all `@staticmethod`)**

* **`sample_stim_durations(order, stimuli_durations, t_pre, t_post)`** → `list[float]`
    * Samples concrete per-trial stimulus durations for a given order, using the distribution specs in `stimuli_durations`.
    * Adds `t_pre + t_post` to each trial's duration.
    * Supports `"fixed"`, `"exponential"`, `"uniform"`, `"gaussian"` distribution models, as well as scalar values.

* **`generate_iti(order, conditional_iti)`** → `list[float]`
    * Samples a concrete ITI array based on condition-dependent distributions.
    * For the first trial (no previous stimulus), uses the `"default"` key.
    * Supports `"fixed"`, `"exponential"`, `"uniform"`, `"gaussian"` distribution models.
    * Returns a list of length `len(order)`.

* **`calculate_duration(ITI, dur)`** → `float`
    * Computes total duration as the sum of all ITIs and all trial durations.

* **`sample_from_probabilities(prob, key, length)`** → `list`
    * Generates a stimulus order by sampling `length` times from `key` with weights `prob`, then flattening the sampled sequences into a single order list.

---

## 3. Modified Functions

### **Design Class**

* **`__init__(self, order, ITI, experiment, onsets=None, all_stim_durations=None)`**
    * **Added:** `all_stim_durations` parameter for per-trial variable stimulus durations.

* **`designmatrix(self)`**
    * Added branches for variable stimulus durations via `self.all_stim_durations`. When set, onset computation and design matrix construction use per-trial durations instead of a single `stim_duration`.
    * Returns `False` if the design's actual timing exceeds the Experiment container, allowing the optimizer to skip invalid candidates gracefully.

* **`crossover(self, other, seed)`**
    * Added fixed-order support: when `order_fixed=True`, offspring inherit the parent's order unchanged.
    * Propagates `all_stim_durations` to offspring. When the order changes and `stimuli_durations` is set, durations are re-sampled to match the new order. When `order_fixed=True`, durations are inherited.

* **`mutation(self, q, seed)`**
    * Added fixed-order support: mutation is skipped when `order_fixed=True`.
    * Same `all_stim_durations` propagation logic as crossover.

### **Experiment Class**

* **`countstim(self)`**
    * Duration is computed from expected values (`n_trials × (trial_duration + ITImean)`) regardless of whether `stimuli_durations` is set, ensuring the whitening matrix dimensions are stable across all designs in a population.

* **`max_eff(self)`**
    * Extended to estimate FeMax and FdMax by sampling random designs and taking the maximum raw efficiency. This provides initial normalization for Fe and Fd even when evaluating designs manually (outside the optimizer). Respects user-provided values.

### **Optimisation Class**

* **`add_new_designs(self, weights, R)`**
    * Added order sampling paths: fixed order → `self.exp.order`; probability-based → `Experiment.sample_from_probabilities()`; default → `generate.order()` (original behavior).
    * Added ITI sampling paths: conditional → `Experiment.generate_iti()`; default → `generate.iti()` (original behavior).
    * Added stimulus duration sampling: when `stimuli_durations` is set, calls `Experiment.sample_stim_durations()` per design.
    * All designs share a single Experiment object, ensuring efficiency metrics remain on a comparable scale.

* **`clear(self)`**
    * Propagates `all_stim_durations` when preserving the best design across generation clears.

* **`optimise(self)`**
    * Skips Fe/Fd calibration pre-runs when the corresponding weight is zero or when the user has provided explicit FeMax/FdMax values.

* **`to_next_generation(self, weights, seed, optimisation)`**
    * When `order_fixed=True`, skips mutation and crossover entirely, using immigration only. This focuses optimization on ITI timing and other non-order parameters.
