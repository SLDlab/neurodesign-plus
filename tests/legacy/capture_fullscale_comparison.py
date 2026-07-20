"""One-time capture script for a full-scale GA optimisation comparison.

Compares neurodesign and neurodesign_legacy on a typical, legacy-compatible
flat experimental design. Not a pytest test (tests/legacy/ is excluded from
pytest collection via norecursedirs). Run manually to regenerate the frozen
fixture at tests/fixtures/legacy_parity/fullscale_comparison.json if the
reference comparison ever needs to be recaptured.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = REPO_ROOT / "tests" / "legacy"
if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))

from neurodesign_legacy import (  # noqa: E402
    Experiment as LegacyExperiment,
    Optimisation as LegacyOptimisation,
)

from neurodesign import Experiment, Optimisation  # noqa: E402

FIXTURE_PATH = (
    REPO_ROOT / "tests" / "fixtures" / "legacy_parity" / "fullscale_comparison.json"
)

# A typical flat experimental design: 24 trials, 3 conditions, fixed ITI.
# Expressible identically in both the live and legacy Experiment APIs.
N_TRIALS = 24
N_STIMULI = 3
STIM_DURATION = 1.0
T_PRE = 0.5
T_POST = 0.2
ITI_MEAN = 2.0
TR = 2.0
DURATION = N_TRIALS * (STIM_DURATION + T_PRE + T_POST + ITI_MEAN)

# Settings reused from a previously accepted, known-feasible "nontrivial"
# optimisation run in this repo's manuscript-support evidence
# (validation/manuscript_support/generate_case10_manuscript_figures.py
# used a smaller toy variant; this repeats the larger nontrivial variant
# recorded in .suplex/docs/08_status_checkpoint.md, 2026-07-04).
SEED = 20260716
PRERUNCYCLES = 2
CYCLES = 4
G = 12
I = 4
OUTDES = 3
# Fe/Fd weighted at zero: neurodesign_legacy's own FdCalc (np.matrix.trace)
# is incompatible with the current NumPy version and crashes inside its own
# optimise() loop otherwise. Same limitation already documented and worked
# around in tests/test_legacy_parity.py's end-to-end test.
WEIGHTS = [0.0, 0.0, 0.5, 0.5]


def build_experiments():
    new_exp = Experiment(
        TR=TR,
        n_trials=N_TRIALS,
        duration=DURATION,
        P=[1 / 3, 1 / 3, 1 / 3],
        C=[[1, 0, -1], [0, 1, -1], [1, -1, 0]],
        n_stimuli=N_STIMULI,
        rho=0.3,
        event_durations=STIM_DURATION,
        trial_start_interval=T_PRE,
        post_event_interval=T_POST,
        inter_trial_interval=ITI_MEAN,
        resolution=0.1,
        seed=SEED,
    )
    legacy_exp = LegacyExperiment(
        TR=TR,
        n_trials=N_TRIALS,
        P=[1 / 3, 1 / 3, 1 / 3],
        C=[[1, 0, -1], [0, 1, -1], [1, -1, 0]],
        rho=0.3,
        stim_duration=STIM_DURATION,
        n_stimuli=N_STIMULI,
        ITImodel="fixed",
        ITImean=ITI_MEAN,
        t_pre=T_PRE,
        t_post=T_POST,
        resolution=0.1,
    )
    return new_exp, legacy_exp


def run_optimisation(opt_cls, experiment):
    start = time.perf_counter()
    opt = opt_cls(
        experiment=experiment,
        weights=WEIGHTS,
        preruncycles=PRERUNCYCLES,
        cycles=CYCLES,
        seed=SEED,
        optimisation="GA",
        G=G,
        I=I,
        outdes=OUTDES,
        convergence=1000,
    )
    opt.optimise()
    elapsed = time.perf_counter() - start
    best = opt.bestdesign
    return {
        "runtime_seconds": elapsed,
        "generations_completed": getattr(opt, "generations_completed", None),
        "best_order": list(best.order),
        "F": float(best.F),
        "Fe": float(best.Fe),
        "Fd": float(best.Fd),
        "Ff": float(best.Ff),
        "Fc": float(best.Fc),
    }


def main():
    new_exp, legacy_exp = build_experiments()

    print("Running live neurodesign optimisation...")
    new_result = run_optimisation(Optimisation, new_exp)
    print(f"  done in {new_result['runtime_seconds']:.1f}s")

    print("Running neurodesign_legacy optimisation...")
    legacy_result = run_optimisation(LegacyOptimisation, legacy_exp)
    print(f"  done in {legacy_result['runtime_seconds']:.1f}s")

    payload = {
        "config": {
            "n_trials": N_TRIALS,
            "n_stimuli": N_STIMULI,
            "stim_duration": STIM_DURATION,
            "t_pre": T_PRE,
            "t_post": T_POST,
            "iti_mean": ITI_MEAN,
            "TR": TR,
            "duration": DURATION,
            "seed": SEED,
            "preruncycles": PRERUNCYCLES,
            "cycles": CYCLES,
            "G": G,
            "I": I,
            "outdes": OUTDES,
            "weights": WEIGHTS,
        },
        "neurodesign": new_result,
        "neurodesign_legacy": legacy_result,
    }
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
