"""One-time capture script: modest multi-config, multi-seed GA comparison.

This script compares ``neurodesign`` and ``neurodesign_legacy`` on a shared
flat-design subset that both APIs can express directly. It records each
package's own normalized metrics after re-scoring the winning design, plus
independent raw/common-scale statistics for Fe/Fd/Ff/Fc.

Not a pytest test (``tests/legacy/`` is excluded from pytest collection).
Run manually to regenerate the frozen fixture at
``tests/fixtures/legacy_parity/sweep_comparison.json`` and the corresponding
log under ``tests/fixtures/legacy_parity/run_logs/`` when this evidence needs
to be recaptured.
"""

from __future__ import annotations

import copy
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.linalg

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = REPO_ROOT / "tests" / "legacy"
if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from neurodesign_legacy import (  # noqa: E402
    Experiment as LegacyExperiment,
    Optimisation as LegacyOptimisation,
)

from neurodesign import Experiment, Optimisation  # noqa: E402

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "legacy_parity"
FIXTURE_PATH = FIXTURE_DIR / "sweep_comparison.json"
LOG_PATH = FIXTURE_DIR / "run_logs" / "sweep_comparison_run.log"

TR = 2.0
STIM_DURATION = 1.0
T_PRE = 0.5
T_POST = 0.2
ITI_MEAN = 2.0
RHO = 0.3
RESOLUTION = 0.1
PRERUNCYCLES = 1
CYCLES = 1
G = 2
I = 1
OUTDES = 1
CONVERGENCE = 10
WEIGHTS = [0.25, 0.25, 0.25, 0.25]
CONFIGS = [
    {"config_id": "trials_12_stimuli_2", "n_trials": 12, "n_stimuli": 2},
    {"config_id": "trials_14_stimuli_2", "n_trials": 14, "n_stimuli": 2},
    {"config_id": "trials_16_stimuli_2", "n_trials": 16, "n_stimuli": 2},
]
SEEDS = [101, 202, 303]


class Tee:
    """Mirror writes to multiple text streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def legacy_compatible_duration(n_trials: int) -> float:
    return n_trials * (STIM_DURATION + T_PRE + T_POST + ITI_MEAN)


def equal_probabilities(n_stimuli: int) -> list[float]:
    return [1.0 / n_stimuli] * n_stimuli


def default_contrasts(n_stimuli: int) -> list[list[int]]:
    return [
        [1 if i == stim else -1 if i == n_stimuli - 1 else 0 for i in range(n_stimuli)]
        for stim in range(n_stimuli - 1)
    ]


def build_experiments(config: dict[str, int], seed: int):
    n_trials = config["n_trials"]
    n_stimuli = config["n_stimuli"]
    probabilities = equal_probabilities(n_stimuli)
    contrasts = default_contrasts(n_stimuli)
    duration = legacy_compatible_duration(n_trials)

    new_exp = Experiment(
        TR=TR,
        n_trials=n_trials,
        duration=duration,
        P=probabilities,
        C=contrasts,
        rho=RHO,
        n_stimuli=n_stimuli,
        event_durations=STIM_DURATION,
        trial_start_interval=T_PRE,
        post_event_interval=T_POST,
        inter_trial_interval=ITI_MEAN,
        resolution=RESOLUTION,
        seed=seed,
    )
    legacy_exp = LegacyExperiment(
        TR=TR,
        n_trials=n_trials,
        P=probabilities,
        C=contrasts,
        rho=RHO,
        stim_duration=STIM_DURATION,
        n_stimuli=n_stimuli,
        ITImodel="fixed",
        ITImean=ITI_MEAN,
        t_pre=T_PRE,
        t_post=T_POST,
        resolution=RESOLUTION,
    )
    return new_exp, legacy_exp


def scalarize(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def estimation_raw(design) -> float:
    inv_m = np.array(scipy.linalg.pinv(design.X))
    cmc = np.dot(np.dot(design.CX, inv_m), np.transpose(design.CX))
    return scalarize(design.CX.shape[0] / np.trace(cmc))


def detection_raw(design) -> float:
    inv_m = np.array(scipy.linalg.pinv(design.Z))
    cmc = np.array(design.C) @ inv_m @ np.transpose(np.array(design.C))
    return scalarize(len(design.C) / np.trace(cmc))


def frequency_mismatch_raw(design) -> float:
    observed = np.array(
        [Counter(design.order).get(x, 0) for x in range(design.experiment.n_stimuli)],
        dtype=float,
    )
    expected = float(len(design.order)) * np.array(design.experiment.P, dtype=float)
    return float(np.sum(np.abs(observed - expected)))


def transition_mismatch_raw(design, confoundorder: int) -> float:
    q = np.zeros(
        (design.experiment.n_stimuli, design.experiment.n_stimuli, confoundorder)
    )
    for n in range(len(design.order)):
        for r in np.arange(1, confoundorder + 1):
            if n > (r - 1):
                q[design.order[n], design.order[n - r], r - 1] += 1

    qexp = np.zeros_like(q)
    for si in range(design.experiment.n_stimuli):
        for sj in range(design.experiment.n_stimuli):
            for r in np.arange(1, confoundorder + 1):
                qexp[si, sj, r - 1] = (
                    design.experiment.P[si]
                    * design.experiment.P[sj]
                    * (len(design.order) + 1)
                )

    return float(np.sum(np.abs(q - qexp)))


def rescore_live_design(opt: Optimisation):
    design = copy.deepcopy(opt.selected_design(0))
    design.designmatrix().FCalc(weights=WEIGHTS, confoundorder=opt.exp.confoundorder)
    return design


def rescore_legacy_design(opt: LegacyOptimisation):
    design = copy.deepcopy(opt.bestdesign)
    design.designmatrix()
    design.FCalc(weights=np.array(WEIGHTS), confoundorder=opt.exp.confoundorder)
    return design


def capture_result(
    package_name: str, opt_cls, experiment, seed: int
) -> dict[str, object]:
    print(f"Running {package_name} optimisation for seed={seed}...")
    start = time.perf_counter()
    opt = opt_cls(
        experiment=experiment,
        weights=WEIGHTS,
        preruncycles=PRERUNCYCLES,
        cycles=CYCLES,
        seed=seed,
        optimisation="GA",
        G=G,
        I=I,
        outdes=OUTDES,
        convergence=CONVERGENCE,
    )
    opt.optimise()
    elapsed = time.perf_counter() - start

    design = (
        rescore_live_design(opt)
        if package_name == "neurodesign"
        else rescore_legacy_design(opt)
    )
    confoundorder = design.experiment.confoundorder
    raw_metrics = {
        "Fe_raw_a_optimality": estimation_raw(design),
        "Fd_raw_a_optimality": detection_raw(design),
        "Ff_raw_frequency_mismatch": frequency_mismatch_raw(design),
        "Fc_raw_transition_mismatch": transition_mismatch_raw(design, confoundorder),
    }
    normalized_metrics = {
        "F": float(design.F),
        "Fe": float(design.Fe),
        "Fd": float(design.Fd),
        "Ff": float(design.Ff),
        "Fc": float(design.Fc),
    }
    exp = design.experiment
    result = {
        "runtime_seconds": elapsed,
        "best_order": [int(x) for x in design.order],
        "normalized_metrics": normalized_metrics,
        "raw_common_scale_metrics": raw_metrics,
        "normalization_references": {
            "FeMax": float(exp.FeMax),
            "FdMax": float(exp.FdMax),
            "FfMax": float(exp.FfMax),
            "FcMax": float(exp.FcMax),
            "confoundorder": int(exp.confoundorder),
        },
    }
    print(
        "  finished in "
        f"{elapsed:.1f}s; normalized F={normalized_metrics['F']:.6f}, "
        f"raw Fe={raw_metrics['Fe_raw_a_optimality']:.6f}, "
        f"raw Fd={raw_metrics['Fd_raw_a_optimality']:.6f}, "
        f"raw Ff mismatch={raw_metrics['Ff_raw_frequency_mismatch']:.6f}, "
        f"raw Fc mismatch={raw_metrics['Fc_raw_transition_mismatch']:.6f}"
    )
    return result


def config_payload(config: dict[str, int]) -> dict[str, object]:
    n_trials = config["n_trials"]
    n_stimuli = config["n_stimuli"]
    return {
        **config,
        "TR": TR,
        "duration": legacy_compatible_duration(n_trials),
        "stim_duration": STIM_DURATION,
        "t_pre": T_PRE,
        "t_post": T_POST,
        "iti_mean": ITI_MEAN,
        "rho": RHO,
        "resolution": RESOLUTION,
        "P": equal_probabilities(n_stimuli),
        "C": default_contrasts(n_stimuli),
    }


def main():
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()
    with LOG_PATH.open("w", encoding="utf-8") as log_file:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(sys.stdout, log_file)
        sys.stderr = Tee(sys.stderr, log_file)
        try:
            print("Starting sweep comparison capture")
            print(f"Python: {platform.python_version()} ({sys.executable})")
            print(f"Platform: {platform.platform()}")
            print(
                "Settings: "
                f"preruncycles={PRERUNCYCLES}, cycles={CYCLES}, G={G}, I={I}, "
                f"outdes={OUTDES}, convergence={CONVERGENCE}, weights={WEIGHTS}"
            )
            print(f"Configs: {[cfg['config_id'] for cfg in CONFIGS]}")
            print(f"Seeds: {SEEDS}")

            runs = []
            for config in CONFIGS:
                print()
                print(f"=== Config {config['config_id']} ===")
                for seed in SEEDS:
                    pair_start = time.perf_counter()
                    new_exp, legacy_exp = build_experiments(config, seed)
                    neurodesign_result = capture_result(
                        "neurodesign", Optimisation, new_exp, seed
                    )
                    legacy_result = capture_result(
                        "neurodesign_legacy", LegacyOptimisation, legacy_exp, seed
                    )
                    pair_elapsed = time.perf_counter() - pair_start
                    print(f"Pair wall time: {pair_elapsed:.1f}s")
                    runs.append(
                        {
                            "config": config_payload(config),
                            "seed": seed,
                            "pair_runtime_seconds": pair_elapsed,
                            "neurodesign": neurodesign_result,
                            "neurodesign_legacy": legacy_result,
                        }
                    )

            total_elapsed = time.perf_counter() - total_start
            payload = {
                "metadata": {
                    "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "python_version": platform.python_version(),
                    "python_executable": sys.executable,
                    "platform": platform.platform(),
                    "log_path": str(LOG_PATH.relative_to(REPO_ROOT)),
                    "total_wall_clock_seconds": total_elapsed,
                },
                "settings": {
                    "preruncycles": PRERUNCYCLES,
                    "cycles": CYCLES,
                    "G": G,
                    "I": I,
                    "outdes": OUTDES,
                    "convergence": CONVERGENCE,
                    "weights": WEIGHTS,
                },
                "runs": runs,
            }
            FIXTURE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print()
            print(f"Wrote {FIXTURE_PATH}")
            print(f"Total wall time: {total_elapsed:.1f}s")
        finally:
            sys.stdout = stdout
            sys.stderr = stderr


if __name__ == "__main__":
    main()
