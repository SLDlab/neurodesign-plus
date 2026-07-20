from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from neurodesign import Experiment, Optimisation, report
from neurodesign.classes import normalize_rule, sample_normalized_rule

REPO_ROOT = Path(__file__).resolve().parents[1]
V1_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "v1_reference"


def _patched_fcalc(field, value):
    def inner(self, *args, **kwargs):
        self.Fe = 1.0
        self.Fd = 1.0
        self.Ff = 1.0
        self.Fc = 1.0
        self.F = 1.0
        setattr(self, field, value)

    return inner


def _real_convergence_population(convergence):
    exp = Experiment(
        TR=2.0,
        n_trials=4,
        P=[0.5, 0.5],
        C=[[1, -1]],
        n_stimuli=2,
        rho=0.3,
        order=[0, 1, 0, 1],
        event_durations=1.0,
        trial_start_interval=0.0,
        post_event_interval=0.0,
        inter_trial_interval=0.0,
        resolution=0.1,
        seed=7,
    )
    return Optimisation(
        experiment=exp,
        weights=[0.0, 0.0, 0.25, 0.25],
        preruncycles=1,
        cycles=4,
        seed=123,
        optimisation="simulation",
        G=2,
        I=1,
        outdes=1,
        convergence=convergence,
    )


def test_hardprob_aligns_counts_by_category_index():
    exp = Experiment(
        TR=1.0,
        n_trials=6,
        P=[1 / 3, 1 / 3, 1 / 3],
        C=[[1, 0, -1]],
        n_stimuli=3,
        rho=0.3,
        order=[2, 2, 0, 1, 0, 1],
        event_durations=1.0,
        trial_start_interval=0.0,
        post_event_interval=0.0,
        inter_trial_interval=0.0,
        hardprob=True,
        resolution=0.1,
        seed=1,
    )
    design = exp.create_design(seed=1)
    assert design.check_hardprob() is True


def test_hardprob_rejects_absent_categories_and_invalid_codes():
    exp = Experiment(
        TR=1.0,
        n_trials=3,
        P=[1 / 3, 1 / 3, 1 / 3],
        C=[[1, 0, -1]],
        n_stimuli=3,
        rho=0.3,
        order=[0, 0, 1],
        event_durations=1.0,
        trial_start_interval=0.0,
        post_event_interval=0.0,
        inter_trial_interval=0.0,
        resolution=0.1,
        seed=2,
    )
    design = exp.create_design(seed=2)
    assert design.check_hardprob() is False
    design.order[0] = 99
    assert design.check_hardprob() is False


def test_hardprob_rejects_probability_length_mismatch():
    exp = Experiment(
        TR=1.0,
        n_trials=3,
        P=[0.5, 0.5],
        C=[[1, -1]],
        n_stimuli=2,
        rho=0.3,
        order=[0, 1, 0],
        event_durations=1.0,
        trial_start_interval=0.0,
        post_event_interval=0.0,
        inter_trial_interval=0.0,
        resolution=0.1,
        seed=3,
    )
    design = exp.create_design(seed=3)
    design.experiment.P = np.array([1 / 3, 1 / 3, 1 / 3])
    assert design.check_hardprob() is False


@pytest.mark.parametrize(
    ("field", "value"),
    [("Fe", np.nan), ("Fd", np.inf), ("Fc", -np.inf), ("F", np.nan)],
)
def test_check_develop_rejects_non_finite_scores(
    monkeypatch, case10_experiment, field, value
):
    design = case10_experiment.create_design(seed=12)
    population = Optimisation(
        experiment=case10_experiment,
        weights=[0.0, 0.5, 0.25, 0.25],
        preruncycles=1,
        cycles=1,
        seed=10,
        optimisation="simulation",
        G=2,
        I=1,
        outdes=1,
    )
    monkeypatch.setattr(type(design), "FCalc", _patched_fcalc(field, value), raising=True)
    assert population.check_develop(design) is False


def test_candidate_generation_fails_boundedly(monkeypatch, scalar_experiment):
    population = Optimisation(
        experiment=scalar_experiment,
        weights=[0.0, 0.5, 0.25, 0.25],
        preruncycles=1,
        cycles=1,
        seed=10,
        optimisation="simulation",
        G=1,
        I=1,
        outdes=1,
        max_candidate_attempts=3,
    )
    monkeypatch.setattr(population, "check_develop", lambda design, weights=None: False)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        population.add_new_designs(R=[1])


def test_real_convergence_example_stops_after_requested_stagnation():
    population = _real_convergence_population(convergence=1)
    population.optimise()
    design = population.selected_design(0)
    assert population.finished is True
    assert population.generations_completed == 2
    assert "1 consecutive generation" in population.stop_reason
    assert population.optima == pytest.approx([design.F, design.F])
    assert design.F == pytest.approx(population.bestscore)


def test_real_convergence_example_records_best_generation():
    population = _real_convergence_population(convergence=1)
    population.optimise()
    assert population.bestdesign_generation == 1
    assert population.bestdesign is not None


def test_real_convergence_example_can_be_disabled():
    population = _real_convergence_population(convergence=None)
    population.optimise()
    design = population.selected_design(0)
    assert population.generations_completed == 4
    assert population.stop_reason is None
    assert population.optima == pytest.approx([design.F, design.F, design.F, design.F])


def test_optimise_breaks_when_finished(monkeypatch, scalar_experiment):
    population = Optimisation(
        experiment=scalar_experiment,
        weights=[0.0, 0.0, 0.25, 0.25],
        preruncycles=1,
        cycles=5,
        seed=12,
        optimisation="simulation",
        G=1,
        I=1,
        outdes=1,
    )
    calls = {"count": 0}
    monkeypatch.setattr(population, "add_new_designs", lambda *args, **kwargs: population)
    monkeypatch.setattr(population, "clear", lambda: population)

    def fake_to_next_generation(*args, **kwargs):
        calls["count"] += 1
        population.generations_completed = calls["count"]
        population.finished = calls["count"] >= 2
        return population

    monkeypatch.setattr(population, "to_next_generation", fake_to_next_generation)
    population.optimise()
    assert calls["count"] == 2


@pytest.mark.parametrize(
    "spec",
    [
        {"model": "gaussian", "mean": 1.4, "std": 0.3, "min": 0.5, "max": 2.5},
        {"model": "exponential", "mean": 1.6, "min": 0.5, "max": 3.0},
    ],
)
def test_bounded_sampling_stays_within_bounds_and_matches_mean(spec):
    rng = np.random.default_rng(4)
    rule = normalize_rule(spec, "timing")
    samples = np.array(
        [sample_normalized_rule(rule, "timing", rng, 0.01) for _ in range(4000)]
    )
    assert np.all(samples >= spec["min"])
    assert np.all(samples <= spec["max"])
    assert abs(samples.mean() - spec["mean"]) < 0.08
    lower_mass = np.mean(np.isclose(samples, spec["min"]))
    upper_mass = np.mean(np.isclose(samples, spec["max"]))
    assert lower_mass < 0.1
    assert upper_mass < 0.1


def test_invalid_bounded_distribution_parameters_raise():
    with pytest.raises(ValueError, match="outside bounds"):
        normalize_rule(
            {"model": "exponential", "mean": 5.0, "min": 0.5, "max": 3.0}, "iti"
        )
    with pytest.raises(ValueError, match="std must be finite and > 0"):
        normalize_rule(
            {"model": "gaussian", "mean": 1.0, "std": 0.0, "min": 0.5, "max": 2.0}, "dur"
        )
    with pytest.raises(ValueError, match="mean must equal"):
        normalize_rule({"model": "uniform", "min": 1.0, "max": 3.0, "mean": 2.5}, "iti")
    with pytest.raises(ValueError, match="finite"):
        normalize_rule(
            {
                "model": "gaussian",
                "mean": 1.0,
                "std": float("inf"),
                "min": 0.5,
                "max": 2.0,
            },
            "dur",
        )


def test_negative_unbounded_duration_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        Experiment(
            TR=1.0,
            n_trials=4,
            P=[1.0],
            C=[[1]],
            n_stimuli=1,
            rho=0.3,
            event_durations=-0.5,
        )


def test_report_handles_seven_event_categories_without_three_condition_assumptions(
    case10_experiment,
    tmp_path,
    monkeypatch,
):
    population = Optimisation(
        experiment=case10_experiment,
        weights=[0.0, 0.5, 0.25, 0.25],
        preruncycles=1,
        cycles=1,
        folder=tmp_path,
        seed=100,
        G=2,
        I=1,
        outdes=1,
        optimisation="simulation",
    )
    population.optimise()
    population.evaluate()
    captured_shapes = []
    original_corrcoef = report.np.corrcoef

    def spy_corrcoef(array, *args, **kwargs):
        out = original_corrcoef(array, *args, **kwargs)
        if (
            getattr(array, "ndim", 0) == 2
            and array.shape[0] == case10_experiment.n_stimuli
        ):
            captured_shapes.append(out.shape)
        return out

    monkeypatch.setattr(report.np, "corrcoef", spy_corrcoef)
    report.make_report(population, tmp_path / "report.pdf")
    assert (case10_experiment.n_stimuli, case10_experiment.n_stimuli) in captured_shapes
    source = inspect.getsource(report.make_report)
    assert "Regressor corr." in source
    assert "range(3)" not in source


def test_selected_design_requires_optimise_first(case10_experiment):
    population = Optimisation(
        experiment=case10_experiment,
        weights=[0.0, 0.5, 0.25, 0.25],
        preruncycles=1,
        cycles=1,
        seed=100,
        G=2,
        I=1,
        outdes=1,
        optimisation="simulation",
    )
    with pytest.raises(RuntimeError, match="requires optimise\\(\\)"):
        population.selected_design(0)


def test_selected_design_invalid_rank_raises_index_error(case10_experiment):
    population = Optimisation(
        experiment=case10_experiment,
        weights=[0.0, 0.5, 0.25, 0.25],
        preruncycles=1,
        cycles=1,
        seed=100,
        G=2,
        I=1,
        outdes=1,
        optimisation="simulation",
    )
    population.optimise()
    with pytest.raises(IndexError, match="out of range"):
        population.selected_design(1)


def test_readme_public_workflow_runs(tmp_path):
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
    population = Optimisation(
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
        folder=tmp_path / "readme_example",
    )
    population.optimise()
    design = population.selected_design(0)
    report_path = tmp_path / "report.pdf"
    schedule_path = tmp_path / "schedule.json"
    spec_path = tmp_path / "specification.json"
    report.make_report(population, report_path)
    schedule_path.write_text(
        json.dumps(design.export_payload(), indent=2), encoding="utf-8"
    )
    spec_path.write_text(
        json.dumps(exp.export_specification(), indent=2), encoding="utf-8"
    )
    assert report_path.exists()
    assert schedule_path.exists()
    assert spec_path.exists()


def test_resolution_adjustment_warning_uses_valid_warning_api():
    with pytest.warns(UserWarning, match="resolution is adjusted"):
        Experiment(
            TR=2.0,
            n_trials=4,
            P=[0.5, 0.5],
            C=[[1, -1]],
            n_stimuli=2,
            rho=0.3,
            event_durations=1.0,
            trial_start_interval=0.0,
            post_event_interval=0.0,
            inter_trial_interval=0.0,
            resolution=0.3,
            seed=1,
        )


def test_case10_manuscript_provenance_hashes_match(tmp_path):
    env = os.environ.copy()
    env["NEURODESIGN_VALIDATION_MANUSCRIPT_OUTPUT"] = str(tmp_path)
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "validation.manuscript_support.generate_case10_manuscript_figures",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    provenance = json.loads(
        (tmp_path / "case10_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["equality_status"]["all_equal"] is True


def test_one_event_scalar_matches_stable_v1_reference():
    fixture = json.loads(
        (V1_FIXTURES / "one_event_scalar_reference.json").read_text(encoding="utf-8")
    )
    exp = Experiment(
        TR=2.0,
        n_trials=4,
        P=[0.5, 0.5],
        C=[[1, -1]],
        n_stimuli=2,
        rho=0.3,
        order=[0, 1, 0, 1],
        event_durations=1.0,
        trial_start_interval=0.5,
        post_event_interval=0.2,
        inter_trial_interval=2.0,
        resolution=0.1,
        seed=7,
        confoundorder=1,
    )
    design = exp.create_design(seed=7)
    design.designmatrix().FCalc(weights=[0.0, 0.5, 0.25, 0.25], confoundorder=1)

    assert design.order == fixture["order"]
    np.testing.assert_allclose(design.event_onsets, fixture["event_onsets"])
    np.testing.assert_allclose(design.event_offsets, fixture["event_offsets"])
    np.testing.assert_allclose(design.Xnonconv, fixture["xnonconv"])
    np.testing.assert_allclose(design.Xconv, fixture["xconv"])
    assert design.Fe == pytest.approx(fixture["metrics"]["Fe"])
    assert design.Fd == pytest.approx(fixture["metrics"]["Fd"])
    assert design.Ff == pytest.approx(fixture["metrics"]["Ff"])
    assert design.Fc == pytest.approx(fixture["metrics"]["Fc"])
