# Technical Documentation: Refactoring Details

This document outlines the specific internal modifications made to the `classes.py` file in the Neurodesign package. It details new class variables, added functions, and modifications to existing logic within the `Design`, `Experiment`, and `Optimisation` classes.

---

## Architecture Overview

The refactored code enforces a strict separation between the **Experiment** (a fixed container defining the experimental context) and the **Design** (a specific trial sequence within that container).

- **Experiment** stores distribution *specifications* (ITI model parameters, stimulus duration specs) and computes the whitening matrix, timepoints, and HRF components once. It does not hold per-trial timing arrays.
- **Design** stores concrete per-trial arrays (order, ITI, stimulus durations) sampled from the Experiment's specifications. All designs in an optimization population share a single Experiment object.

This guarantees that all efficiency metrics (Fe, Fd) are computed against the same whitening matrix, making them comparable across designs.

---

## 1. Added Class Variables

### **Design Class**

* **`all_stim_durations`** `(list of floats or None)`
    * Concrete per-trial stimulus durations for this design, including `t_pre` and `t_post`.
    * `None` when all stimuli share the same `stim_duration` (base package behavior).
    * When set, `designmatrix()` uses these values instead of `experiment.stim_duration`.
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
    * Length must equal `n_stimuli`. If not provided, all stimuli use `stim_duration`.
    * **This is a template only** — concrete per-trial arrays are sampled per-Design via `Experiment.sample_stim_durations()`.

* **`conditional_ITI`** `(dict or None)`
    * Specification for condition-dependent ITI distributions. Keys are `(prev_stim, curr_stim)` tuples or `"default"`. Values are dicts with `"model"`, `"mean"`, and optional `"min"`, `"max"`, `"std"`.
    * **Example:**
        ```python
        conditional_ITI = {
            (0, 1): {"model": "exponential", "mean": 2, "min": 1},
            (1, 2): {"model": "fixed", "mean": 4},
            "default": {"model": "exponential", "mean": 3, "min": 1}
        }
        ```
    * **This is a template only** — concrete ITI arrays are sampled per-Design via `Experiment.generate_iti()`.

* **`order`** `(list of ints or None)`
    * A user-provided fixed stimulus order. When provided, `order_fixed` is set to `True`.
    * Used for all designs during optimization (crossover and mutation preserve this order).

* **`trial_max`** `(float)`
    * The maximum stimulus duration across all conditions. Defaults to `stim_duration` when `stimuli_durations` is not provided.
    * Used to compute the container's `trial_duration = trial_max + t_pre + t_post`, which determines the experiment's total duration and thus the whitening matrix dimensions.

#### **Sequence Generation Variables**

* **`order_keys`** `(list of integer lists)` — Sequences of stimuli used as sampling units.
* **`order_probabilities`** `(list of floats)` — Probabilities for each key (must sum to 1).
* **`order_length`** `(int)` — Number of draws from the key distribution.

These three are inputs for `sample_from_probabilities()`.

---

## 2. New Functions

### **Experiment Class (all `@staticmethod`)**

All new functions on Experiment are static methods. They operate on their arguments without reading or modifying Experiment state, reinforcing the principle that per-trial randomness lives on the Design, not the Experiment.

* **`sample_stim_durations(order, stimuli_durations, t_pre, t_post)`** → `list[float]`
    * Samples concrete per-trial stimulus durations for a given order, using the distribution specs in `stimuli_durations`.
    * Adds `t_pre + t_post` to each trial's duration.
    * Supports `"fixed"`, `"exponential"`, `"uniform"`, `"gaussian"` models, and scalar values.
    * Called once per Design in `add_new_designs()`, `crossover()`, and `mutation()`.
    * *Replaces the former instance method `calculate_all_stimuli()` which modified Experiment state.*

* **`generate_iti(order, conditional_iti)`** → `list[float]`
    * Samples a concrete ITI array based on condition-dependent distributions.
    * For the first trial (no previous stimulus), uses the `"default"` key.
    * Supports `"fixed"`, `"exponential"`, `"uniform"`, `"gaussian"` models.
    * Returns a list of length `len(order)`.
    * *Formerly an instance method. The gaussian branch bug (appending to `self.all_stim_durations` instead of `ITI`) has been fixed.*

* **`calculate_duration(ITI, dur)`** → `float`
    * Computes total duration as the sum of all ITIs and all trial durations.
    * Utility function; not used during standard Experiment construction (which uses expected-value computation instead).

* **`sample_from_probabilities(prob, key, length)`** → `list`
    * Generates a stimulus order by sampling `length` times from `key` with weights `prob`, then flattening.
    * *Formerly an instance method; logic unchanged.*

---

## 3. Modified Functions

### **Design Class**

* **`__init__(self, order, ITI, experiment, onsets=None, all_stim_durations=None)`**
    * **Added:** `all_stim_durations` parameter. Stored as `self.all_stim_durations`.
    * **Removed:** The override `if self.experiment.ITI is not None: self.ITI = self.experiment.ITI`. This was silently replacing every Design's ITI (including the internal NulDesign used for calibration), which corrupted Fc/Ff normalization.

* **`designmatrix(self)`**
    * **Changed:** Reads `self.all_stim_durations` (from the Design) instead of `self.experiment.all_stim_durations`.
    * **Changed:** Hard `assert` statements for timepoint bounds replaced with a soft overflow check. Returns `False` if the design's actual timing exceeds the Experiment container. This prevents crashes from extreme ITI draws (exponential tail).
    * **Fixed:** Rest-block indexing bug. When `restnum > 0` and variable durations are used, rest blocks expand `orderli` with `"R"` entries, but `all_stim_durations` only has `n_trials` entries. The original code indexed `all_stim_durations[i]` with the expanded position. Fixed by introducing a separate `trial_idx` counter that only increments for non-rest trials.

* **`crossover(self, other, seed)`**
    * **Added:** Propagates `all_stim_durations` to offspring. When `order_fixed=False` and `stimuli_durations` is set, durations are re-sampled via `Experiment.sample_stim_durations()` to match the new order. When `order_fixed=True`, durations are inherited from the parent.
    * Fixed order behavior is unchanged: offspring receive the parent's order.

* **`mutation(self, q, seed)`**
    * **Added:** Same `all_stim_durations` propagation logic as crossover. Re-samples when order changes; inherits when order is fixed.
    * Fixed order behavior is unchanged: mutation is skipped.

### **Experiment Class**

* **`__init__`**
    * **Removed:** `ITI` parameter from the constructor signature. Concrete ITI arrays are no longer stored on Experiment.
    * **Removed:** ITI generation logic (calls to `generate.iti()` and `self.generate_iti()` that previously ran during construction).
    * **Added:** `_user_FeMax` and `_user_FdMax` boolean flags. Track whether the user explicitly provided FeMax/FdMax values. Used by `optimise()` to decide whether to run the calibration pre-run.
    * Stores `conditional_ITI` directly (simplified from the previous if/else block).
    * Stores `order` and calls `sample_from_probabilities()` if `order_probabilities` is provided.

* **`countstim(self)`**
    * **Simplified:** Removed the `stimuli_durations` branch that called `calculate_all_stimuli()` and `calculate_duration()`. Duration is now always computed from expected values: `n_trials × (trial_duration + ITImean)`, where `trial_duration = trial_max + t_pre + t_post`. This ensures the whitening matrix dimensions are stable across all designs.

* **`max_eff(self)`**
    * **Extended:** In addition to calibrating FcMax and FfMax via the NulDesign (unchanged), now also estimates FeMax and FdMax by sampling 100 random designs and taking the maximum raw efficiency. This provides a reasonable initial normalization even when using metrics outside of the optimizer.
    * Only calibrates Fe/Fd when they are still at the default value of 1 (respects user-provided values and optimizer pre-run overrides).

### **Optimisation Class**

* **`add_new_designs(self, weights, R)`**
    * **Rewritten.** The previous version created a new `Experiment` object for every candidate design when `conditional_ITI` or `order_probabilities` was set. This broke the normalization invariant (different whitening matrix per design).
    * **Now:** All designs use `self.exp`. The method samples order, ITI, and stimulus durations independently, then passes them to `Design(order=..., ITI=..., experiment=self.exp, all_stim_durations=...)`.
    * Order sampling: fixed order → `self.exp.order`; probability-based → `Experiment.sample_from_probabilities()`; default → `generate.order()`.
    * ITI sampling: conditional → `Experiment.generate_iti()`; default → `generate.iti()`.
    * Stim duration sampling: if `stimuli_durations` is set → `Experiment.sample_stim_durations()`; otherwise `None`.
    * **Also fixed:** `t_pre=0, t_post=0` were previously hardcoded in the per-design Experiment constructor. Eliminated by removing per-design Experiment creation.

* **`clear(self)`**
    * **Changed:** Propagates `all_stim_durations` when preserving the best design across clears.

* **`optimise(self)`**
    * **Changed:** Uses `_user_FeMax` / `_user_FdMax` flags to skip the pre-run calibration when the user explicitly provided values or when `max_eff()` already set reasonable estimates. Also skips Fe pre-run when `weights[0] == 0` and Fd pre-run when `weights[1] == 0`.

* **`to_next_generation(self, weights, seed, optimisation)`**
    * **Unchanged** from previous version. When `order_fixed=True`, skips mutation and crossover, using immigration only.
