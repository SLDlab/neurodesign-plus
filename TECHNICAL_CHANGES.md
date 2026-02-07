# Technical Documentation: Refactoring Details

This document outlines the specific internal modifications made to the `classes.py` file in the Neurodesign package. It details new class variables, added functions, and modifications to existing logic within the `Design`, `Experiment`, and `Optimisation` classes.

## 1. Added Class Variables

### **Experiment Class**

* **`stimuli_durations`** `(list of int or dicts)`
    * A list of `stimuli_durations` corresponding to each stimulus (length of `stimuli_durations` must be the same as `n_stimuli`).
    * If not provided, `stimuli_durations` uses `stimuli_duration` (the previous class duration variable) for all stimuli.

* **`conditional_iti`** `(dict)`
    * Adds a new input of ITI that can be calculated from the chosen probability distribution of the users. This variable holds the information about the probability distribution for an ITI between two specific stimuli.
    * There is also a `'default'` key used if a relationship between two specific ITI in order is not provided.
    * **Example Structure:**
        ```python
        self.conditional_ITI = {
            (0, 1): {"model": "exponential", "mean": 2, "min": 1},
            (1, 2): {"model": "fixed", "mean": 4},
            "default": {"model": "exponential", "mean": 3, "min": 1}
        }
        ```

* **`order`** `(list of ints or None)`
    * Allows the user to input their own custom order for their experiment.
    * If `order` is provided, `self.order_fixed` (boolean) is set to **True**. This means the order is fixed and will be used for all designs and optimization.

* **`ITI`** `(list of floats)`
    * Allows the user to input their own custom ITI. This ITI can be modified later with the `conditional_iti` or the `ITImodel` provided.
    * Acts as the baseline ITI for the model.

* **`trial_max`** `(float)`
    * Refers to the max `stimuli_duration` provided in the list of `stimuli_durations`.
    * Used to calculate a maximum `n_tp` (timepoints), so the XConv calculation in the design can work for all varying ITIs and stimuli durations.

* **`all_stimuli_durations`** `(list of floats)`
    * Based on the given `stimuli_durations` provided, calculates the duration of each stimulus in order. This is used in the `Design` class in all cases when `stimuli_durations` have been provided.

#### **Sequence Generation Variables**
*The following three variables are used to generate an order with keys (sequences). They are inputs for the `sample_from_probabilities` function.*

* **`order_keys`** `(list of integer lists)`
    * The list of sequences corresponding to stimuli/stimuli ordering.
* **`order_probabilities`** `(list of floats)`
    * Defines the probability of the corresponding sequences (must sum to 1).
* **`order_length`** `(int)`
    * The number of times we want to select from the `order_keys` based on the `order_probabilities`.

---

## 2. New Functions

### **Experiment Class**

* **`calculate_duration(self, ITI, dur)`**
    * Generates/calculates the duration based on `stimuli_duration` and ITI.

* **`sample_from_probabilities(self, prob, key, length)`**
    * Generates an order based on a given probability distribution of certain keys (sequences of stimuli).

* **`generate_iti(self, order, conditional_iti)`**
    * Generates ITI with the first value using the `'default'` entry, then appended relative to distributions of stimuli in order.
    * Ensures `n_trials` and ITI length are the same.

* **`calculate_all_stimuli(self)`**
    * Calculates the `all_stimuli_durations` based on the new order (also calculates duration).

---

## 3. Modified Functions

### **Design Class**

* **`design_matrix`**
    * Added cases for multiple stimuli durations by working with `all_stimuli_durations`.
    * If only `stim_duration` is provided, it uses the base code of the package.

* **`mutation`**
    * Added a case for **fixed order**, making the mutation utilize the same order for any new design.

* **`crossover`**
    * Added a case for **fixed order**, making the offspring orders the same as the parent order.

### **Experiment Class**

* **`count_stim`**
    * Added cases for when `stimuli_durations` (multiple stimuli) is provided.
    * When there are various durations, order matters; therefore, this function now accounts for cases where a fixed order is provided or not.

### **Optimisation Class**

* **`add_new_designs`**
    * Added handling for the case of **fixed order**, custom order probabilities/functions, and default package order generator.
    * Added handling for the case of custom ITI probabilities/functions and default package ITI generator.

* **`to_next_generation`**
    * When there is a **fixed order**, mutation and crossover should not run because the focus is on optimizing the order.
    * In this case, the function only uses the immigration optimization function.