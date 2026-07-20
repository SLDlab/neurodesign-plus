from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg

from neurodesign import Experiment, Optimisation, msequence

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = REPO_ROOT / "tests" / "legacy"
if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".tmp_mpl"))
os.environ.setdefault("IPYTHONDIR", str(REPO_ROOT / ".tmp_ipython"))

from neurodesign_legacy import (  # noqa: E402
    Design as LegacyDesign,
    Experiment as LegacyExperiment,
    Optimisation as LegacyOptimisation,
    msequence_legacy,
)


def _legacy_compatible_duration(
    n_trials: int, stim_duration: float, t_pre: float, t_post: float, iti_mean: float
) -> float:
    return n_trials * (stim_duration + t_pre + t_post + iti_mean)


def _build_flat_experiments():
    n_trials = 12
    stim_duration = 1.0
    t_pre = 0.5
    t_post = 0.2
    iti_mean = 2.0
    duration = _legacy_compatible_duration(
        n_trials, stim_duration, t_pre, t_post, iti_mean
    )
    new_exp = Experiment(
        TR=2.0,
        n_trials=n_trials,
        duration=duration,
        P=[0.5, 0.5],
        C=[[1, -1]],
        n_stimuli=2,
        rho=0.3,
        event_durations=stim_duration,
        trial_start_interval=t_pre,
        post_event_interval=t_post,
        inter_trial_interval=iti_mean,
        resolution=0.1,
        seed=101,
    )
    legacy_exp = LegacyExperiment(
        TR=2.0,
        n_trials=n_trials,
        P=[0.5, 0.5],
        C=[[1, -1]],
        rho=0.3,
        stim_duration=stim_duration,
        n_stimuli=2,
        ITImodel="fixed",
        ITImean=iti_mean,
        t_pre=t_pre,
        t_post=t_post,
        resolution=0.1,
    )
    return new_exp, legacy_exp


def _build_manual_designs():
    new_exp, legacy_exp = _build_flat_experiments()
    order = [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1]
    iti = [0.0] + [2.0] * (len(order) - 1)
    new_design = new_exp.create_manual_design(
        order=order,
        inter_trial_intervals=iti,
        event_durations=[1.0] * len(order),
    )
    legacy_design = LegacyDesign(order=order, ITI=iti, experiment=legacy_exp)
    return new_exp, legacy_exp, new_design, legacy_design


def _legacy_scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _legacy_fe(design: LegacyDesign) -> float:
    inv_m = np.array(scipy.linalg.pinv(design.X))
    cmc = np.dot(np.dot(design.CX, inv_m), np.transpose(design.CX))
    return _legacy_scalar(design.CX.shape[0] / np.trace(cmc)) / design.experiment.FeMax


def _legacy_fd(design: LegacyDesign) -> float:
    inv_m = np.array(scipy.linalg.pinv(design.Z))
    cmc = np.matrix(design.C) * inv_m * np.matrix(np.transpose(design.C))
    return _legacy_scalar(len(design.C) / np.trace(cmc)) / design.experiment.FdMax


def _legacy_ff(design: LegacyDesign) -> float:
    trialcount = Counter(design.order)
    observed = [trialcount[x] for x in range(design.experiment.n_stimuli)]
    mismatch = np.sum(
        abs(
            np.array(observed)
            - np.array(design.experiment.n_trials * np.array(design.experiment.P))
        )
    )
    return 1 - mismatch / design.experiment.FfMax


def _legacy_fc(design: LegacyDesign, confoundorder: int = 1) -> float:
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
                    * (design.experiment.n_trials + 1)
                )
    mismatch = np.sum(abs(q - qexp))
    return 1 - mismatch / design.experiment.FcMax


def test_legacy_msequence_generation_matches_exactly():
    new_order = msequence.Msequence().GenMseq(mLen=25, stimtypeno=4, seed=42).orders[0]
    legacy_order = (
        msequence_legacy.Msequence().GenMseq(mLen=25, stimtypeno=4, seed=42).orders[0]
    )

    assert new_order == legacy_order


def test_legacy_metric_calculations_match_on_fixed_design_matrix():
    _, _, new_design, legacy_design = _build_manual_designs()
    new_design.designmatrix().FCalc(weights=[0.25, 0.25, 0.25, 0.25], confoundorder=1)
    legacy_design.designmatrix()

    np.testing.assert_allclose(new_design.Xnonconv, legacy_design.Xnonconv)
    np.testing.assert_allclose(new_design.Xconv, legacy_design.Xconv, atol=0.01, rtol=0.0)
    assert new_design.Fe == pytest.approx(_legacy_fe(legacy_design), rel=0.03)
    assert new_design.Fd == pytest.approx(_legacy_fd(legacy_design), abs=5e-4)
    assert new_design.Ff == pytest.approx(_legacy_ff(legacy_design))
    assert new_design._transition_mismatch(1) == pytest.approx(7.0)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "v2.0 recalibrates FcMax for the requested confound order, while legacy "
        "reuses a constructor-time FcMax normalized at its default confoundorder=3"
    ),
)
def test_legacy_fc_normalization_matches_on_fixed_design_matrix():
    _, _, new_design, legacy_design = _build_manual_designs()
    new_design.designmatrix().FcCalc(confoundorder=1)
    legacy_design.designmatrix()

    assert new_design.Fc == pytest.approx(_legacy_fc(legacy_design, confoundorder=1))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "v2.0 replaced legacy global NumPy reseeding with SeedSequence/Generator RNGs, "
        "so fixed-seed mutation no longer reproduces the legacy offspring order exactly"
    ),
)
def test_legacy_ga_mutation_matches_under_fixed_seed():
    _, _, new_design, legacy_design = _build_manual_designs()

    mutated_new = new_design.mutation(0.25, seed=11)
    mutated_legacy = legacy_design.mutation(0.25, seed=11)

    assert mutated_new.order == list(mutated_legacy.order)


def test_legacy_short_end_to_end_ga_run_matches_best_score():
    new_exp, legacy_exp = _build_flat_experiments()
    weights = [0.0, 0.0, 0.5, 0.5]
    new_opt = Optimisation(
        experiment=new_exp,
        weights=weights,
        preruncycles=1,
        cycles=1,
        seed=101,
        optimisation="GA",
        G=4,
        I=2,
        outdes=1,
        convergence=10,
    )
    legacy_opt = LegacyOptimisation(
        experiment=legacy_exp,
        weights=np.array(weights),
        preruncycles=1,
        cycles=1,
        seed=101,
        optimisation="GA",
        G=4,
        I=2,
        outdes=1,
        convergence=10,
    )

    new_opt.clear()
    new_opt.add_new_designs()
    new_opt.to_next_generation(seed=101)

    legacy_opt.clear()
    legacy_opt.add_new_designs()
    legacy_opt.to_next_generation(seed=101)

    assert new_opt.bestdesign.F == pytest.approx(legacy_opt.bestdesign.F)
    assert new_opt.bestdesign.Ff == pytest.approx(legacy_opt.bestdesign.Ff)
    assert new_opt.bestdesign.Fc == pytest.approx(legacy_opt.bestdesign.Fc)
