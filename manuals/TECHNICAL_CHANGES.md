# Technical Documentation: Refactoring Details

This document outlines the specific internal modifications made to `classes.py` in `neurodesign-plus`. It focuses on the new class variables, helper methods, and behavior changes in the `Design`, `Experiment`, and `Optimisation` classes.

---

## Architecture Overview

The refactored code preserves a strict separation between the **Experiment** and the **Design**:

- **Experiment** stores the fixed specification of the task container: HRF settings, whitening matrix inputs, ITI summaries, and duration specifications.
- **Design** stores one concrete sampled realization: event order, ITIs, and optionally per-event sampled stimulus durations.

This separation is important because all designs in a population share the same experiment container and therefore the same whitening matrix. That keeps raw Fe and Fd values comparable across designs generated within that experiment.

---

## 1. Added Class Variables

### **Design Class**

* **`all_stim_durations`** `(list of floats or None)`
    * Concrete per-trial stimulus durations for this design, including `t_pre` and `t_post`.
    * `None` when all stimuli share the same `stim_duration`.
    * When set, `designmatrix()` uses these per-trial values instead of `experiment.stim_duration`.
    * Sampled via `Experiment.sample_stim_durations()` and passed at construction: `Design(order=..., ITI=..., experiment=..., all_stim_durations=...)`.

### **Experiment Class**

* **`stimuli_durations`** `(list of int/float or dicts, or None)`
    * A specification template -- one entry per stimulus -- defining how to sample each stimulus's duration.
    * Each entry is either a scalar (fixed duration) or a dict specifying a distribution:
        ```python
        stimuli_durations = [
            {"model": "fixed", "mean": 1.0},
            {"model": "exponential", "mean": 2.0, "min": 1.0, "max": 5.0},
            1.5,
        ]
        ```
    * Length must equal `n_stimuli`.
    * Requires `trial_max` to be specified.
    * If not provided, all stimuli use the original `stim_duration`.

* **`conditional_ITI`** `(dict or None)`
    * Specification for transition-dependent ITI distributions.
    * Keys are `(prev_stim, curr_stim)` tuples or `"default"`.
    * Values are dicts with `"model"`, `"mean"`, and optional `"min"`, `"max"`, `"std"`.
    * Example:
        ```python
        conditional_ITI = {
            (0, 1): {"model": "exponential", "mean": 2, "min": 1},
            (1, 2): {"model": "fixed", "mean": 4},
            "default": {"model": "exponential", "mean": 3, "min": 1},
        }
        ```

* **`order`** `(list of ints or None)`
    * A user-provided fixed event order.
    * When provided, `order_fixed` is set to `True`.
    * The order is preserved across all designs during optimization -- crossover and mutation do not modify it.

* **`trial_max`** `(float)`
    * The maximum raw stimulus duration across all conditions.
    * Required when `stimuli_durations` is provided.
    * Used to compute the experiment container's `trial_duration = trial_max + t_pre + t_post`.

#### **Sequence Generation Variables**

* **`order_keys`** `(list of integer lists)` -- Sequences sampled as units.
* **`order_probabilities`** `(list of floats)` -- Probabilities for each key.
* **`order_length`** `(int)` -- Requested length of the final event-level order.

These three are inputs for `sample_from_probabilities()`. When `order_probabilities` is provided, the optimizer generates event orders by sampling from these sequence templates rather than using the original blocked, random, or m-sequence generators. Internally, the sampled templates are flattened and then truncated to the first `order_length` events.

---

## 2. New Functions

### **Experiment Class (all `@staticmethod`)**

* **`sample_stim_durations(order, stimuli_durations, t_pre, t_post)`** -> `list[float]`
    * Samples concrete per-trial stimulus durations for a given order, using the specifications in `stimuli_durations`.
    * Adds `t_pre + t_post` to each sampled duration.
    * Supports `"fixed"`, `"exponential"`, `"uniform"`, and `"gaussian"` models, as well as scalar values.

* **`generate_iti(order, conditional_iti)`** -> `list[float]`
    * Samples a concrete ITI array from a transition-dependent ITI specification.
    * Uses the `"default"` key for the first event and for any transition not explicitly listed.
    * Supports `"fixed"`, `"exponential"`, `"uniform"`, and `"gaussian"` models.
    * Returns a list of length `len(order)`.

* **`calculate_duration(ITI, dur)`** -> `float`
    * Computes total duration as the sum of all ITIs and all trial durations.

* **`sample_from_probabilities(prob, key, length)`** -> `list`
    * Samples templates from `key` with weights `prob`.
    * Flattens the sampled templates into one event-level order.
    * Returns the first `length` events from that flattened order.
    * Because the final list is truncated to `length`, the last sampled template can be cut at the tail when templates have different lengths.

---

## 3. Modified Functions

### **Design Class**

* **`__init__(self, order, ITI, experiment, onsets=None, all_stim_durations=None)`**
    * Added `all_stim_durations` for per-trial variable stimulus durations.

* **`designmatrix(self)`**
    * Added support for variable stimulus durations via `self.all_stim_durations`.
    * When set, onset computation and design matrix construction use per-trial durations instead of a single shared `stim_duration`.
    * Returns `False` if the design's actual timing exceeds the experiment container.

* **`crossover(self, other, seed)`**
    * Added fixed-order support: when `order_fixed=True`, offspring inherit the parent orders unchanged.
    * When `stimuli_durations` is set and the order changes, per-trial durations are re-sampled to match the new order.
    * When `order_fixed=True`, per-trial durations are inherited.

* **`mutation(self, q, seed)`**
    * Added fixed-order support: mutation only changes the order when `order_fixed=False` and `order_probabilities is None`.
    * Uses the same per-trial duration propagation logic as crossover.

### **Experiment Class**

* **`countstim(self)`**
    * Computes the container duration from expected values: `n_trials x (trial_duration + ITImean)`.
    * This behavior is kept even when `stimuli_durations` is set, so the whitening matrix dimensions stay stable across all designs in a population.

* **`max_eff(self)`**
    * Computes `FcMax` and `FfMax` from a null design built inside the shared experiment container.
    * The current implementation does not estimate `FeMax` or `FdMax` inside `max_eff()`.

### **Optimisation Class**

* **`add_new_designs(self, weights, R)`**
    * Added order sampling paths:
      `fixed order -> self.exp.order`
      `probability-based order -> Experiment.sample_from_probabilities()`
      `default -> generate.order()`
    * Added ITI sampling paths:
      `conditional_ITI -> Experiment.generate_iti()`
      `default -> generate.iti()`
    * Added per-design sampling of stimulus durations when `stimuli_durations` is set.
    * Keeps all generated designs attached to one shared experiment object.

* **`clear(self)`**
    * Propagates `all_stim_durations` when preserving the best design across generation clears.

* **`optimise(self)`**
    * Skips Fe/Fd calibration pre-runs when the corresponding weight is zero or when the user has provided explicit FeMax/FdMax values.

* **`to_next_generation(self, weights, seed, optimisation)`**
    * When `order_fixed=True`, skips mutation and crossover and uses immigration only.
    * This focuses optimization on timing and other non-order parameters.
