import json

import numpy as np
import pytest

from neurodesign import Design, Experiment, Optimisation
from neurodesign.classes import (
    Design as ClassesDesign,
    Experiment as ClassesExperiment,
    Optimisation as ClassesOptimisation,
)
from validation.helpers.case10_spec import TRIAL_TEMPLATES, build_case10_experiment


def test_authoritative_public_imports():
    assert Experiment is ClassesExperiment
    assert Design is ClassesDesign
    assert Optimisation is ClassesOptimisation


def test_removed_and_ambiguous_inputs_error():
    with pytest.raises(TypeError, match="trial_start_interval"):
        Experiment(
            TR=2.0,
            n_trials=4,
            P=[0.5, 0.5],
            C=[[1, -1]],
            n_stimuli=2,
            rho=0.3,
            event_durations=1.0,
            t_pre=0.5,
        )
    with pytest.raises(TypeError, match="event_durations"):
        Experiment(
            TR=2.0,
            n_trials=4,
            P=[0.5, 0.5],
            C=[[1, -1]],
            n_stimuli=2,
            rho=0.3,
            stimuli_durations=1.0,
        )


def test_uncovered_selector_errors(multi_event_templates):
    exp = Experiment(
        TR=1.0,
        P=[1 / 7] * 7,
        C=[[1, 0, 0, 0, 0, 0, -1]],
        rho=0.3,
        n_stimuli=7,
        trial_templates=multi_event_templates,
        trials=[{"template_id": "hint_branch"}],
        event_durations=1.0,
        trial_start_interval={"by_trial_type": {"standard": 0.0}},
        post_event_interval=0.0,
        event_transition_interval=0.0,
        inter_trial_interval=1.0,
        resolution=0.1,
        seed=1,
    )
    with pytest.raises(ValueError, match="no rule for selector"):
        exp.create_design()


def test_negative_and_non_finite_duration_errors():
    with pytest.raises(ValueError, match="non-negative"):
        Experiment(
            TR=2.0,
            n_trials=4,
            P=[1.0],
            C=[[1]],
            n_stimuli=1,
            rho=0.3,
            event_durations=-1.0,
        )
    with pytest.raises(ValueError, match="finite"):
        Experiment(
            TR=2.0,
            n_trials=4,
            P=[1.0],
            C=[[1]],
            n_stimuli=1,
            rho=0.3,
            event_durations={
                "model": "gaussian",
                "mean": 1.0,
                "std": float("inf"),
                "min": 0.0,
            },
        )

    with pytest.raises(ValueError, match="ambiguous"):
        Experiment(
            TR=2.0,
            n_trials=4,
            P=[0.5, 0.5],
            C=[[1, -1]],
            n_stimuli=2,
            rho=0.3,
            event_durations=1.0,
            inter_trial_interval={"min": 1.0, "max": 2.0},
        )


def test_one_event_scalar_equivalence(scalar_experiment):
    design_a = scalar_experiment.create_manual_design(
        order=[0, 1, 0, 1],
        inter_trial_intervals=[2.0, 2.0, 2.0, 2.0],
        event_durations=[1.0, 1.0, 1.0, 1.0],
    )
    design_b = scalar_experiment.create_manual_design(
        order=[0, 1, 0, 1],
        inter_trial_intervals=[2.0, 2.0, 2.0, 2.0],
        event_durations=[1.0, 1.0, 1.0, 1.0],
    )
    design_a.designmatrix().FCalc(weights=[0.0, 0.5, 0.25, 0.25])
    design_b.designmatrix().FCalc(weights=[0.0, 0.5, 0.25, 0.25])

    np.testing.assert_allclose(design_a.event_onsets, design_b.event_onsets)
    np.testing.assert_allclose(
        design_a.realized_event_durations, design_b.realized_event_durations
    )
    np.testing.assert_allclose(design_a.Xnonconv, design_b.Xnonconv)
    np.testing.assert_allclose(design_a.Xconv, design_b.Xconv)
    np.testing.assert_allclose(
        [design_a.Fe, design_a.Fd, design_a.Ff, design_a.Fc],
        [design_b.Fe, design_b.Fd, design_b.Ff, design_b.Fc],
    )


def test_multi_event_schedule_exact(multi_event_templates):
    exp = Experiment(
        TR=1.0,
        P=[1 / 7] * 7,
        C=[[1, 0, 0, 0, 0, 0, -1]],
        rho=0.3,
        n_stimuli=7,
        trial_templates=multi_event_templates,
        trials=[{"template_id": "standard"}, {"template_id": "hint_branch"}],
        trial_start_interval=0.5,
        post_event_interval=0.2,
        event_transition_interval={
            "by_event_transition": {
                ("cue_easy", "choice_left"): 0.4,
                ("choice_left", "feedback"): 0.6,
                ("cue_hard", "hint"): 0.1,
                ("hint", "choice_right"): 0.3,
                ("choice_right", "feedback"): 0.5,
            }
        },
        inter_trial_interval=1.5,
        event_durations=1.0,
        resolution=0.1,
        seed=5,
    )
    design = exp.create_design()
    np.testing.assert_allclose(design.trial_starts, [0.0, 6.6])
    np.testing.assert_allclose(
        design.event_onsets,
        [0.5, 1.9, 3.9, 7.1, 8.2, 9.4, 11.3],
    )
    np.testing.assert_allclose(design.trial_ends, [5.1, 12.5])


def test_xnonconv_excludes_all_intervals(multi_event_templates):
    exp = Experiment(
        TR=1.0,
        P=[1 / 7] * 7,
        C=[[1, 0, 0, 0, 0, 0, -1]],
        rho=0.3,
        n_stimuli=7,
        trial_templates=multi_event_templates,
        trials=[{"template_id": "standard"}],
        trial_start_interval=1.0,
        post_event_interval=0.5,
        event_transition_interval={
            "by_event_transition": {
                ("cue_easy", "choice_left"): 0.4,
                ("choice_left", "feedback"): 0.6,
            }
        },
        inter_trial_interval=0.0,
        event_durations=1.0,
        resolution=0.1,
        seed=6,
    )
    design = exp.create_design()
    design.designmatrix()
    total_tp = design.Xnonconv.sum()
    expected_tp = int(round(design.realized_event_durations.sum() / exp.TR))
    assert total_tp == expected_tp
    assert design.event_onsets[0] == pytest.approx(1.0)


def test_event_duration_sampling_by_occurrence():
    exp = Experiment(
        TR=1.0,
        n_trials=4,
        P=[1.0],
        C=[[1]],
        n_stimuli=1,
        rho=0.3,
        event_durations={"model": "uniform", "min": 1.0, "max": 2.0},
        trial_start_interval=0.0,
        post_event_interval=0.0,
        inter_trial_interval=1.0,
        resolution=0.1,
        seed=21,
    )
    design = exp.create_design()
    assert len(np.unique(design.realized_event_durations)) > 1


def test_selector_rules_and_default_fallbacks(multi_event_templates):
    exp = Experiment(
        TR=1.0,
        P=[1 / 7] * 7,
        C=[[1, 0, 0, 0, 0, 0, -1]],
        rho=0.3,
        n_stimuli=7,
        trial_templates=multi_event_templates,
        trials=[{"template_id": "standard"}, {"template_id": "hold_branch"}],
        trial_start_interval={"by_trial_type": {"standard": 0.5, "default": 1.0}},
        post_event_interval={"by_event_category": {"feedback": 0.3, "default": 0.1}},
        event_transition_interval={
            "by_event_transition": {("cue_easy", "choice_left"): 0.4, "default": 0.2}
        },
        inter_trial_interval=1.0,
        event_durations=1.0,
        resolution=0.1,
        seed=10,
    )
    design = exp.create_design()
    np.testing.assert_allclose(design.realized_trial_start_intervals, [0.5, 1.0])
    assert design.realized_post_event_intervals[-1] == pytest.approx(0.3)
    assert set(design.selector_provenance["event_transition_rule_ids"])


def test_no_transition_interval_across_trial_boundary(multi_event_templates):
    exp = Experiment(
        TR=1.0,
        P=[1 / 7] * 7,
        C=[[1, 0, 0, 0, 0, 0, -1]],
        rho=0.3,
        n_stimuli=7,
        trial_templates=multi_event_templates,
        trials=[{"template_id": "standard"}, {"template_id": "standard"}],
        event_durations=1.0,
        trial_start_interval=0.0,
        post_event_interval=0.0,
        event_transition_interval={
            "by_event_transition": {
                ("cue_easy", "choice_left"): 0.4,
                ("choice_left", "feedback"): 0.6,
            }
        },
        inter_trial_interval=1.2,
        resolution=0.1,
        seed=8,
    )
    design = exp.create_design()
    assert len(design.realized_event_transition_intervals) == 4
    assert np.all(
        design.within_trial_transition_to_event_index
        - design.within_trial_transition_from_event_index
        == 1
    )
    assert design.event_onsets[3] - design.event_offsets[2] == pytest.approx(1.2)


def test_complete_template_enforcement(multi_event_templates):
    with pytest.raises(ValueError, match="probabilistic template mode requires"):
        Experiment(
            TR=1.0,
            P=[1 / 7] * 7,
            C=[[1, 0, 0, 0, 0, 0, -1]],
            rho=0.3,
            n_stimuli=7,
            trial_templates=multi_event_templates,
            trial_template_probabilities=[0.5, 0.5, 0.0],
            event_durations=1.0,
        )


def test_flat_order_shorthand_creates_one_event_trials(scalar_experiment):
    design = scalar_experiment.create_design(seed=4)
    assert np.all(design.event_index_within_trial == 0)
    assert len(design.trial_starts) == len(design.order)
    assert len(design.realized_event_transition_intervals) == 0


def test_rest_boundary_timing_additive(multi_event_templates):
    exp = Experiment(
        TR=1.0,
        P=[1 / 7] * 7,
        C=[[1, 0, 0, 0, 0, 0, -1]],
        rho=0.3,
        n_stimuli=7,
        trial_templates=multi_event_templates,
        trials=[
            {"template_id": "standard"},
            {"template_id": "standard"},
            {"template_id": "standard"},
        ],
        event_durations=1.0,
        trial_start_interval=0.5,
        post_event_interval=0.2,
        event_transition_interval={
            "by_event_transition": {
                ("cue_easy", "choice_left"): 0.4,
                ("choice_left", "feedback"): 0.6,
            }
        },
        inter_trial_interval=1.0,
        rest_every_n_trials=2,
        rest_interval=3.0,
        resolution=0.1,
        seed=11,
    )
    design = exp.create_design()
    gap = design.trial_starts[2] - design.trial_ends[1]
    assert gap == pytest.approx(4.0)
    assert design.realized_rest_intervals[1] == pytest.approx(3.0)


def test_seed_reproducibility(case10_experiment):
    design_a = case10_experiment.create_design(seed=12)
    design_b = case10_experiment.create_design(seed=12)
    np.testing.assert_allclose(design_a.event_onsets, design_b.event_onsets)
    np.testing.assert_allclose(
        design_a.realized_event_durations, design_b.realized_event_durations
    )
    assert design_a.trial_template_ids == design_b.trial_template_ids


def test_seed_change_changes_realization(case10_experiment):
    design_a = case10_experiment.create_design(seed=12)
    design_b = case10_experiment.create_design(seed=13)
    assert design_a.trial_template_ids != design_b.trial_template_ids or not np.allclose(
        design_a.realized_event_durations, design_b.realized_event_durations
    )


def test_rng_helpers_do_not_modify_global_state(scalar_experiment):
    before = np.random.get_state()
    scalar_experiment.create_design(seed=10)
    after = np.random.get_state()
    assert before[1].tolist() == after[1].tolist()


def test_export_round_trip_reconstructs_schedule(case10_experiment, tmp_path):
    design = case10_experiment.create_design(seed=12)
    payload = design.export_payload()
    spec = case10_experiment.export_specification()
    schedule_path = tmp_path / "schedule.json"
    spec_path = tmp_path / "spec.json"
    schedule_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    spec_path.write_text(json.dumps(spec, indent=2, default=str), encoding="utf-8")

    loaded = json.loads(schedule_path.read_text())
    np.testing.assert_allclose(
        loaded["schedule_arrays"]["event_onsets"],
        design.event_onsets,
    )
    assert (
        loaded["schedule"][0]["event_category"]
        == design.schedule_table[0]["event_category"]
    )


def test_export_round_trip_preserves_requested_specs(case10_experiment, tmp_path):
    spec = case10_experiment.export_specification()
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    loaded = json.loads(spec_path.read_text())
    assert loaded["inter_trial_interval_requested"]["model"] == "uniform"
    assert loaded["trial_templates"][1]["template_id"] == "hint_branch"


def test_case10_tutorial_and_manuscript_support_share_spec(
    baseline_case10_tutorial,
    baseline_case10_manuscript,
    multi_event_templates,
):
    tutorial_keys = baseline_case10_tutorial["keys"]
    manuscript_keys = baseline_case10_manuscript["experiment_spec"]["order_keys"]
    assert tutorial_keys == manuscript_keys
    assert [template["template_id"] for template in multi_event_templates] == [
        "standard",
        "hint_branch",
        "hold_branch",
    ]
    assert (
        baseline_case10_tutorial["conditional_ITI"]["(6, 0)"]
        == baseline_case10_manuscript["experiment_spec"]["conditional_ITI"]["(6, 0)"]
    )


def test_optimisation_metadata_integrity(case10_experiment, tmp_path):
    population = Optimisation(
        experiment=case10_experiment,
        weights=[0.0, 0.5, 0.25, 0.25],
        preruncycles=1,
        cycles=1,
        folder=tmp_path,
        seed=101,
        optimisation="simulation",
        G=2,
        I=1,
        outdes=1,
    )
    population.optimise()
    design = population.selected_design(0)
    assert len(design.trial_template_ids) == case10_experiment.n_conceptual_trials
    assert len(design.trial_starts) == case10_experiment.n_conceptual_trials
    assert (
        len(design.realized_inter_trial_intervals)
        == case10_experiment.n_conceptual_trials - 1
    )


def test_crossover_metadata_integrity(case10_experiment):
    parent_a = case10_experiment.create_design(seed=12)
    parent_b = case10_experiment.create_design(seed=13)
    child_a, child_b = parent_a.crossover(parent_b, seed=5)
    for child in (child_a, child_b):
        assert len(child.trial_template_ids) == case10_experiment.n_conceptual_trials
        assert len(child.trial_ids) == len(child.order)
        assert max(child.event_index_within_trial) >= 0


def test_mutation_metadata_integrity(case10_experiment):
    design = case10_experiment.create_design(seed=12)
    mutant = design.mutation(0.2, seed=5)
    assert len(mutant.trial_template_ids) == case10_experiment.n_conceptual_trials
    assert len(mutant.trial_ids) == len(mutant.order)
    assert mutant.selector_provenance["event_duration_rule_ids"]


def test_immigration_metadata_integrity(case10_experiment):
    pop = Optimisation(
        experiment=case10_experiment,
        weights=[0.0, 0.5, 0.25, 0.25],
        preruncycles=1,
        cycles=1,
        seed=55,
        optimisation="simulation",
        G=2,
        I=1,
        outdes=1,
    )
    pop.add_new_designs()
    assert pop.designs
    for design in pop.designs:
        assert len(design.trial_template_ids) == case10_experiment.n_conceptual_trials


def test_copying_and_selection_metadata_integrity(case10_experiment):
    original = case10_experiment.create_design(seed=12)
    copied = original.spawn_resampled_timing(case10_experiment.make_design_rng(99))
    assert copied.trial_template_ids == original.trial_template_ids
    assert len(copied.event_index_within_trial) == len(copied.order)


def _transition_mismatch_for_order(order, probabilities, n_stimuli, confoundorder=1):
    observed = np.zeros((n_stimuli, n_stimuli, confoundorder))
    for n in range(len(order)):
        for r in range(1, confoundorder + 1):
            if n > (r - 1):
                observed[order[n], order[n - r], r - 1] += 1
    expected = np.zeros_like(observed)
    for si in range(n_stimuli):
        for sj in range(n_stimuli):
            for r in range(1, confoundorder + 1):
                expected[si, sj, r - 1] = (
                    probabilities[si] * probabilities[sj] * (len(order) + 1)
                )
    return float(np.sum(np.abs(observed - expected)))


def test_case10_v2_matches_v1_ff_and_fc(baseline_case10_tutorial):
    tutorial_trials = [
        {"template_id": TRIAL_TEMPLATES[idx]["template_id"]}
        for idx in baseline_case10_tutorial["sampled_trial_list"]
    ]
    exp = build_case10_experiment(seed=12, trials=tutorial_trials)
    design = exp.create_design(seed=12)
    design.designmatrix().FCalc(weights=[0.0, 0.5, 0.25, 0.25])

    assert design.order == baseline_case10_tutorial["order"]
    assert design.Ff == pytest.approx(0.7071895424836601)
    assert design.Fc == pytest.approx(0.28253393665158355)
    assert design.Ff == pytest.approx(baseline_case10_tutorial["metrics"]["Ff"])
    assert design.Fc == pytest.approx(baseline_case10_tutorial["metrics"]["Fc"])


def test_identical_flattened_order_gives_identical_ff_and_fc_across_representations():
    flat_exp = Experiment(
        TR=1.0,
        P=[0.5, 0.5],
        C=[[1, -1]],
        rho=0.3,
        n_stimuli=2,
        order=[0, 1, 0, 1],
        event_durations=1.0,
        trial_start_interval=0.0,
        post_event_interval=0.0,
        inter_trial_interval=0.0,
        resolution=0.1,
        confoundorder=1,
    )
    flat_design = flat_exp.create_design(seed=1)
    flat_design.designmatrix().FCalc(weights=[0.0, 0.0, 0.5, 0.5], confoundorder=1)

    template_exp = Experiment(
        TR=1.0,
        P=[0.5, 0.5],
        C=[[1, -1]],
        rho=0.3,
        n_stimuli=2,
        trial_templates=[
            {
                "template_id": "ab",
                "trial_type": "ab",
                "events": [
                    {"category": "a", "code": 0, "duration": 1.0},
                    {"category": "b", "code": 1, "duration": 1.0},
                ],
            }
        ],
        trials=[{"template_id": "ab"}, {"template_id": "ab"}],
        event_durations=1.0,
        trial_start_interval=0.0,
        post_event_interval=0.0,
        event_transition_interval=0.0,
        inter_trial_interval=0.0,
        resolution=0.1,
        confoundorder=1,
    )
    template_design = template_exp.create_design(seed=1)
    template_design.designmatrix().FCalc(weights=[0.0, 0.0, 0.5, 0.5], confoundorder=1)

    assert flat_design.order == template_design.order
    assert flat_design.Ff == pytest.approx(template_design.Ff)
    assert flat_design.Fc == pytest.approx(template_design.Fc)


def test_multi_event_design_uses_event_count_for_ff():
    exp = Experiment(
        TR=1.0,
        P=[0.5, 0.5],
        C=[[1, -1]],
        rho=0.3,
        n_stimuli=2,
        trial_templates=[
            {
                "template_id": "ab",
                "trial_type": "ab",
                "events": [
                    {"category": "a", "code": 0, "duration": 1.0},
                    {"category": "b", "code": 1, "duration": 1.0},
                ],
            }
        ],
        trials=[{"template_id": "ab"}, {"template_id": "ab"}],
        event_durations=1.0,
        trial_start_interval=0.0,
        post_event_interval=0.0,
        event_transition_interval=0.0,
        inter_trial_interval=0.0,
        resolution=0.1,
        confoundorder=1,
    )
    design = exp.create_design(seed=1)
    design.designmatrix().FfCalc()

    assert exp.n_conceptual_trials == 2
    assert len(design.order) == 4
    assert design._frequency_mismatch() == pytest.approx(0.0)
    assert design.Ff == pytest.approx(1.0)


def test_fc_expected_counts_use_event_count_not_conceptual_trials():
    exp = Experiment(
        TR=1.0,
        P=[0.5, 0.5],
        C=[[1, -1]],
        rho=0.3,
        n_stimuli=2,
        trial_templates=[
            {
                "template_id": "ab",
                "trial_type": "ab",
                "events": [
                    {"category": "a", "code": 0, "duration": 1.0},
                    {"category": "b", "code": 1, "duration": 1.0},
                ],
            }
        ],
        trials=[{"template_id": "ab"}, {"template_id": "ab"}],
        event_durations=1.0,
        trial_start_interval=0.0,
        post_event_interval=0.0,
        event_transition_interval=0.0,
        inter_trial_interval=0.0,
        resolution=0.1,
        confoundorder=1,
    )
    design = exp.create_design(seed=1)
    expected_with_event_count = _transition_mismatch_for_order(
        design.order, exp.P, exp.n_stimuli, 1
    )
    expected_with_trial_count = _transition_mismatch_for_order(
        design.order[: exp.n_conceptual_trials],
        exp.P,
        exp.n_stimuli,
        1,
    )

    assert len(design.order) == 4
    assert exp.n_conceptual_trials == 2
    assert design._transition_mismatch(1) == pytest.approx(expected_with_event_count)
    assert expected_with_event_count != pytest.approx(expected_with_trial_count)


def test_ffmax_and_fcmax_are_calibrated_on_event_count():
    exp = Experiment(
        TR=1.0,
        P=[0.5, 0.5],
        C=[[1, -1]],
        rho=0.3,
        n_stimuli=2,
        trial_templates=[
            {
                "template_id": "ab",
                "trial_type": "ab",
                "events": [
                    {"category": "a", "code": 0, "duration": 1.0},
                    {"category": "b", "code": 1, "duration": 1.0},
                ],
            }
        ],
        trials=[{"template_id": "ab"}, {"template_id": "ab"}],
        event_durations=1.0,
        trial_start_interval=0.0,
        post_event_interval=0.0,
        event_transition_interval=0.0,
        inter_trial_interval=0.0,
        resolution=0.1,
        confoundorder=1,
    )

    expected_ffmax = 4.0
    expected_fcmax = _transition_mismatch_for_order([0, 0, 0, 0], exp.P, exp.n_stimuli, 1)

    assert exp.FfMax == pytest.approx(expected_ffmax)
    assert exp.FcMax == pytest.approx(expected_fcmax)
    assert exp.ff_max_for_event_count(4) == pytest.approx(expected_ffmax)
    assert exp.fc_max_for_event_count(4, 1) == pytest.approx(expected_fcmax)


def test_optimisation_scoring_prefers_better_event_level_balance():
    exp = Experiment(
        TR=1.0,
        P=[0.5, 0.5],
        C=[[1, -1]],
        rho=0.3,
        n_stimuli=2,
        trial_templates=[
            {
                "template_id": "ab",
                "trial_type": "ab",
                "events": [
                    {"category": "a", "code": 0, "duration": 1.0},
                    {"category": "b", "code": 1, "duration": 1.0},
                ],
            },
            {
                "template_id": "aa",
                "trial_type": "aa",
                "events": [
                    {"category": "a", "code": 0, "duration": 1.0},
                ],
            },
        ],
        trial_template_probabilities=[0.5, 0.5],
        n_conceptual_trials=2,
        event_durations=1.0,
        trial_start_interval=0.0,
        post_event_interval=0.0,
        event_transition_interval=0.0,
        inter_trial_interval=0.0,
        resolution=0.1,
        confoundorder=1,
    )
    balanced_schedule = exp.realize_from_trial_sequence(
        ["ab", "ab"], exp.make_design_rng(1), ["ab", "ab"]
    )
    imbalanced_schedule = exp.realize_from_trial_sequence(
        ["aa", "aa"], exp.make_design_rng(2), ["aa", "aa"]
    )
    balanced = Design(
        experiment=exp,
        schedule=balanced_schedule,
        trial_sequence=["ab", "ab"],
        template_sequence=["ab", "ab"],
    )
    imbalanced = Design(
        experiment=exp,
        schedule=imbalanced_schedule,
        trial_sequence=["aa", "aa"],
        template_sequence=["aa", "aa"],
    )

    balanced.designmatrix().FCalc(weights=[0.0, 0.0, 0.5, 0.5], confoundorder=1)
    imbalanced.designmatrix().FCalc(weights=[0.0, 0.0, 0.5, 0.5], confoundorder=1)

    assert len(balanced.order) == 4
    assert len(imbalanced.order) == 2
    assert balanced.F > imbalanced.F


def test_flat_one_event_design_metrics_remain_unchanged(scalar_experiment):
    design = scalar_experiment.create_design(seed=4)
    design.designmatrix().FCalc(weights=[0.0, 0.0, 0.5, 0.5], confoundorder=1)

    expected_ffmax = scalar_experiment.ff_max_for_event_count(scalar_experiment.n_trials)
    expected_fcmax = scalar_experiment.fc_max_for_event_count(
        scalar_experiment.n_trials, 1
    )
    expected_ff = 1 - design._frequency_mismatch() / expected_ffmax
    expected_fc = 1 - design._transition_mismatch(1) / expected_fcmax

    assert len(design.order) == scalar_experiment.n_trials
    assert design.Ff == pytest.approx(expected_ff)
    assert design.Fc == pytest.approx(expected_fc)


def test_export_payload_and_spec_preserve_both_counts(case10_experiment):
    design = case10_experiment.create_design(seed=12)
    payload = design.export_payload()
    spec = case10_experiment.export_specification()

    assert (
        payload["counts"]["n_conceptual_trials"] == case10_experiment.n_conceptual_trials
    )
    assert payload["counts"]["n_events"] == len(design.order)
    assert spec["n_conceptual_trials"] == case10_experiment.n_conceptual_trials
    assert "n_events" in spec
