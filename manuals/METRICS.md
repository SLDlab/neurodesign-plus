# Neurodesign Efficiency Metrics: A Complete Guide

Neurodesign optimizes fMRI experimental designs by evaluating them on four efficiency metrics. This document explains what each metric measures, how they are computed, how normalization works, and — critically — when you can and cannot compare metric values across different experimental setups.

## Overview

Neurodesign separates the specification of an experiment from the evaluation of candidate designs.

First, define an Experiment: the fixed scientific and acquisition assumptions that determine the “ruler” used to score designs (e.g., TR, total duration or trial budget, autocorrelation parameter rho, target probabilities, contrasts). These settings define how the convolved and deconvolved design matrices are constructed and, for Fe/Fd, fix the whitening matrix 𝑊.

Second, generate and compare Designs: alternative realizations of event timing and ordering that share the same Experiment base. The efficiency metrics then quantify how well each Design performs relative to that base.

A practical consequence is that raw Fe and Fd are meaningful for comparing Designs within the same Experiment, and across Experiments only when 𝑊 is identical (in practice: same duration, TR, and rho). If you change those, you changed the ruler, so the absolute values are on different scales.

---

## The Four Metrics

### Fe — Estimation Efficiency

**What it measures:** How precisely the design allows you to estimate the shape and amplitude of each condition's hemodynamic response, even when events overlap in time.

**Mathematical definition (A-optimality):**

```
Fe_raw = n_contrasts / trace( C × inv(X'WX) × C' )
```

Where:

- `X` is the deconvolved design matrix (stimuli × HRF lag bins), downsampled to TR resolution
- `W` is the whitening matrix (accounts for temporal autocorrelation and drift)
- `C` is the contrast matrix expanded to HRF resolution (`CX = kron(C, I_laghrf)`)
- `trace(C × inv(X'WX) × C')` is the sum of variances of your contrast estimates

**Interpretation:** Fe is the average _precision_ (inverse variance) of your contrast estimates. Higher Fe means the design allows you to more precisely estimate the differences between conditions specified by your contrast matrix. If Design A has twice the Fe of Design B, it means Design A yields contrast estimates with half the variance — you'd need roughly twice as many subjects with Design B to match Design A's precision.

**When it matters most:** Event-related designs where stimuli are closely spaced and temporal overlap is a concern. Fe rewards designs that create orthogonal (non-overlapping) patterns across conditions.

### Fd — Detection Efficiency

**What it measures:** How well the design supports detecting activation differences between conditions, assuming the canonical HRF shape is correct.

**Mathematical definition (A-optimality):**

```
Fd_raw = n_contrasts / trace( C × inv(Z'WZ) × C' )
```

Where `Z` is the _convolved_ design matrix (each column is the stimulus train convolved with the canonical HRF), as opposed to the deconvolved `X` used for Fe.

**Interpretation:** Fd reflects the design's ability to detect that a contrast is nonzero, given that the HRF has the canonical shape. It rewards designs where the convolved regressors for different conditions are maximally distinguishable from each other and from noise.

**When it matters most:** Blocked or mixed designs where you trust the canonical HRF and want to maximize statistical power for detecting activation. In practice, blocked designs tend to have higher Fd than rapid event-related designs.

**Fe vs Fd:** Fe is about _estimating_ the response (flexible HRF shape), Fd is about _detecting_ it (assuming canonical shape). These are sometimes in tension — designs that are optimal for one are often suboptimal for the other, which is why you can weight them differently.

### Fc — Confounding Efficiency

**What it measures:** Whether the sequence of conditions is balanced in terms of transitions — i.e., that no condition systematically follows another condition more than expected by chance.

**Computation:** Fc compares the observed transition matrix (how often condition _i_ follows condition _j_, up to 3rd order) against the expected transition matrix (based on the specified probabilities). The closer the match, the higher Fc.

```
Fc = 1 - |Q_observed - Q_expected| / FcMax
```

Where `FcMax` is calibrated from a worst-case (all-same-stimulus) null design.

**Interpretation:** Fc = 1 means transitions perfectly match what you'd expect from random independent draws at the specified probabilities. Fc = 0 means transitions are maximally biased (e.g., stimulus 0 always follows stimulus 1). Values near 1 are desirable to avoid confounding condition effects with transition effects.

**When it matters most:** Any design where you worry about carry-over effects, adaptation, or expectation. High Fc means the design won't accidentally create systematic patterns (like alternation or repetition) that could confound your contrasts.

### Ff — Frequency Accuracy

**What it measures:** Whether the realized number of trials per condition matches the prescribed probabilities.

**Computation:**

```
Ff = 1 - |P_observed - P_expected| / FfMax
```

**Interpretation:** If you specified P = [0.3, 0.3, 0.4] and got trial counts of [6, 6, 8] out of 20 trials, that's a perfect match (Ff = 1). If you got [10, 5, 5], there's a mismatch (lower Ff). This metric ensures the optimizer doesn't accidentally over- or under-represent conditions.

**When it matters most:** When you have specific requirements about how many trials of each type are needed — for example, to ensure enough power per condition, or when trial counts need to match a specific ratio for your analysis.

---

## Normalization: Raw Values vs Calibrated Values

### The two regimes

Neurodesign metrics can appear in two forms:

**Raw (uncalibrated):** `FeMax = 1`, so `Fe = Fe_raw`. This is what you get when you manually create an Experiment and Designs, then call `FeCalc()`. The values can be any positive number (typically 10–300 for Fe/Fd depending on your experiment).

**Calibrated (normalized):** `FeMax` is set to the best Fe found during an optimization pre-run, so `Fe = Fe_raw / FeMax ∈ [0, 1]`. This is what happens inside `Optimisation.optimise()`.

### How calibration works

When you call `optimise()`, the optimizer:

1. Runs a pre-optimization with `weights=[1, 0, 0, 0]` (only Fe) for `preruncycles` generations
2. Takes the best Fe found and sets `exp.FeMax = best_Fe_raw`
3. Repeats for Fd with `weights=[0, 1, 0, 0]` to calibrate `exp.FdMax`
4. Fc and Ff are calibrated during `Experiment.__init__` using a worst-case null design

After calibration, all four metrics are on a [0, 1] scale, and the weighted sum `F = w₁·Fe + w₂·Fd + w₃·Ff + w₄·Fc` becomes meaningful.

### When to use which

- **Manual design comparison (same Experiment):** Use raw Fe/Fd. They are directly comparable and ratios are meaningful. A design with Fe=200 has exactly 10× the estimation precision of one with Fe=20.

- **During optimization:** Calibrated values are used automatically. You don't need to do anything.

- **Cross-experiment comparison:** See the next section.

---

## Comparing Metrics Across Experiments

### The fundamental rule

**Fe and Fd values are only directly comparable when the whitening matrix `W` is identical.**

The whitening matrix depends on three things:

1. **`n_scans`** = ceil(duration / TR)
2. **`rho`** (autocorrelation coefficient)
3. **Drift polynomials** (derived from `n_scans`)

Therefore: two Experiments with the **same `duration`, `TR`, and `rho`** produce **identical `W`**, and their Fe/Fd values live on the same scale. Everything else (ITI model, stim duration, number of stimuli, contrasts, probabilities) affects the _design matrix_ but not the _ruler_ used to measure it.

### The common trap: specifying n_trials

When you specify `n_trials` instead of `duration`, the code computes:

```
duration = n_trials × (trial_duration + ITImean)
```

If two Experiments have different `ITImean` but the same `n_trials`, they will have **different durations**, **different `n_scans`**, **different `W`**, and their Fe/Fd values are **not comparable**. This is not a bug — it's a fundamental property. A longer experiment gives you more data, which inherently changes the precision you can achieve.

### How to compare ITI models correctly

**Method 1: Fix duration (recommended for comparing design efficiency)**

Specify `duration=` instead of `n_trials=` when creating both Experiments. The duration determines `n_scans` and `W`, which become identical. Differences in ITI model will change the number of trials that fit and the arrangement of events, which is exactly what you're comparing.

```python
# CORRECT: same duration → same W → Fe is comparable
exp_A = Experiment(duration=300, ITImodel="exponential", ITImean=2.1, ...)
exp_B = Experiment(duration=300, ITImodel="uniform", ITImean=3.0, ...)
# exp_A.n_scans == exp_B.n_scans → raw Fe directly comparable
```

**Method 2: Fix n_trials and compare power (recommended for fixed-trial protocols)**

If your experiment must have exactly N trials regardless of timing, then the right comparison is not Fe but _statistical power_ — the probability of detecting your effect at a given significance level and effect size. This accounts for both design efficiency and total data volume.

```python
# ALSO VALID: same n_trials, compare power via simulation
exp_A = Experiment(n_trials=40, ITImodel="exponential", ITImean=2.1, ...)
exp_B = Experiment(n_trials=40, ITImodel="uniform", ITImean=3.0, ...)
# Fe values are NOT comparable, but power analysis gives the right answer
```

### Summary table

| Scenario                                                      | Same `W`? | Raw Fe/Fd comparable? | What to do                                |
| ------------------------------------------------------------- | --------- | --------------------- | ----------------------------------------- |
| Same Experiment, different Design                             | Yes       | Yes                   | Compare raw Fe directly                   |
| Different Experiment, same `duration`/`TR`/`rho`              | Yes       | Yes                   | Compare raw Fe directly                   |
| Different Experiment, same `n_trials` but different `ITImean` | **No**    | **No**                | Use `duration=` instead, or compare power |
| Different `TR`                                                | **No**    | **No**                | Use power analysis                        |
| Different `rho`                                               | **No**    | **No**                | Use power analysis                        |

### Fc and Ff comparisons

Fc and Ff depend only on the stimulus order and probabilities, not on the whitening matrix. They are always comparable across any two designs with the same number of stimuli and the same target probabilities, regardless of timing parameters.

---

## Practical Recommendations

### Choosing weights

The weight vector `[w_Fe, w_Fd, w_Ff, w_Fc]` controls the tradeoff between metrics. Common choices:

- **`[0.25, 0.25, 0.25, 0.25]`**: Balanced — no strong preference. Good starting point.
- **`[1, 0, 0, 0]`**: Pure estimation efficiency. Best when you want to estimate HRF shape freely (e.g., FIR models).
- **`[0, 1, 0, 0]`**: Pure detection power. Best for blocked designs or when you trust the canonical HRF.
- **`[0, 0.5, 0.25, 0.25]`**: Detection-focused with frequency and confound balance. Common for standard GLM analyses.

### Interpreting Fe/Fd magnitudes

Raw (uncalibrated) Fe/Fd values are hard to interpret in isolation. What matters is:

- **Relative ranking:** Within the same Experiment, higher is always better.
- **Ratios:** Fe=200 has 2× the precision of Fe=100. This is linear.
- **Normalized values:** Fe=0.85 (calibrated) means "85% as good as the best design the optimizer found." This tells you how much room there is for improvement.

### When Fe and Fd conflict

For rapid event-related designs, Fe and Fd often point in opposite directions. Jittered ITIs and randomized orders help Fe (by decorrelating the design matrix) but can hurt Fd (by spreading activation over time). If you need both, use mixed weights and accept that neither will be at its theoretical maximum.

### Design matrix sparsity warning

When using variable ITIs (exponential model), some randomly generated designs may produce ITI sequences that cause the total experiment duration to exceed the container. These designs are automatically rejected during optimization. If you see many rejected designs (slow optimization), consider:

1. Using a longer `duration` to give more headroom
2. Using `ITImax` to bound the tail of the exponential distribution
3. Using `trial_max` to set the container size based on the longest possible trial
