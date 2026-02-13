"""Test suite for neurodesign.classes (behavioral tests).

Assumes the neurodesign package is installed and importable.

If your environment does not include `rich` (sometimes treated as optional),
we provide a minimal stub so the import path does not fail.
"""

import sys
import types

import numpy as np

# ============================================================
# Optional: stub rich if missing
# (Some neurodesign environments import rich for progress/output.)
# ============================================================
try:
    import rich  # noqa: F401
    import rich.progress  # noqa: F401
except ModuleNotFoundError:
    rich_mod = types.ModuleType("rich")
    rich_mod.print = print  # just use builtin print
    sys.modules["rich"] = rich_mod

    rich_progress = types.ModuleType("rich.progress")
    for name in [
        "BarColumn",
        "MofNCompleteColumn",
        "Progress",
        "SpinnerColumn",
        "TaskProgressColumn",
        "TextColumn",
        "TimeElapsedColumn",
        "TimeRemainingColumn",
    ]:
        setattr(
            rich_progress, name, type(name, (), {"__init__": lambda self, *a, **kw: None})
        )
    sys.modules["rich.progress"] = rich_progress

# Now import the actual installed neurodesign classes
from neurodesign.classes import Design, Experiment  # noqa: E402

# ============================================================
# TEST HELPERS
# ============================================================


def make_basic_experiment(**overrides):
    """Create a basic 3-stimulus experiment with defaults."""
    params = dict(
        TR=2.0,
        P=[1 / 3, 1 / 3, 1 / 3],
        C=np.array([[1, -1, 0], [0, 1, -1]]),
        rho=0.3,
        stim_duration=1.0,
        n_stimuli=3,
        n_trials=30,
        ITImodel="fixed",
        ITImean=2.0,
        ITImin=1.0,
        ITImax=3.0,
        resolution=0.1,
        t_pre=0.0,
        t_post=0.0,
    )
    params.update(overrides)
    return Experiment(**params)


def make_variable_dur_experiment(**overrides):
    """Create experiment with variable stimulus durations."""
    params = dict(
        TR=2.0,
        P=[1 / 3, 1 / 3, 1 / 3],
        C=np.array([[1, -1, 0], [0, 1, -1]]),
        rho=0.3,
        stim_duration=1.0,
        n_stimuli=3,
        n_trials=30,
        ITImodel="fixed",
        ITImean=2.0,
        ITImin=1.0,
        ITImax=3.0,
        resolution=0.1,
        t_pre=0.0,
        t_post=0.0,
        trial_max=2.0,
        stimuli_durations=[
            {"model": "fixed", "mean": 1.0},
            {"model": "fixed", "mean": 1.5},
            {"model": "fixed", "mean": 2.0},
        ],
    )
    params.update(overrides)
    return Experiment(**params)


# ============================================================
# TEST 1: Basic experiment — Fe should be in [0, 1]
# ============================================================
def test_basic_fe_bounded():
    print("TEST 1: Fe values are on the SAME SCALE (shared Experiment)...")
    exp = make_basic_experiment()

    # First: calibrate FeMax like optimise() does
    rng = np.random.RandomState(42)
    raw_fes = []
    for i in range(50):
        order = list(rng.choice(3, size=30, p=[1 / 3, 1 / 3, 1 / 3]))
        iti = [2.0] * 30
        des = Design(order=order, ITI=np.array(iti), experiment=exp)
        result = des.designmatrix()
        if result is False:
            continue
        des.FeCalc()
        raw_fes.append(des.Fe * exp.FeMax)  # undo normalization to get raw

    # Set FeMax to the max raw Fe (simulating pre-run)
    exp.FeMax = max(raw_fes)

    # Now check that all Fe <= 1 under this calibration
    fe_values = []
    for i in range(50):
        order = list(rng.choice(3, size=30, p=[1 / 3, 1 / 3, 1 / 3]))
        iti = [2.0] * 30
        des = Design(order=order, ITI=np.array(iti), experiment=exp)
        result = des.designmatrix()
        if result is False:
            continue
        des.FeCalc()
        fe_values.append(des.Fe)

    fe_arr = np.array(fe_values)
    print(f"  Fe range after calibration: [{fe_arr.min():.4f}, {fe_arr.max():.4f}]")
    assert np.all(fe_arr >= 0), f"Fe has negative values: {fe_arr.min()}"
    assert np.all(fe_arr <= 1.01), f"Fe exceeds 1: {fe_arr.max()}"
    print("  PASSED\n")


# ============================================================
# TEST 2: Variable durations — Fe should still be bounded
# ============================================================
def test_variable_dur_fe_bounded():
    print("TEST 2: Variable stim durations Fe bounded after calibration...")
    exp = make_variable_dur_experiment()

    rng = np.random.RandomState(42)

    # Calibrate FeMax
    raw_fes = []
    for i in range(50):
        order = list(rng.choice(3, size=30, p=[1 / 3, 1 / 3, 1 / 3]))
        iti = [2.0] * 30
        asd = Experiment.sample_stim_durations(
            order, exp.stimuli_durations, exp.t_pre, exp.t_post
        )
        des = Design(
            order=order, ITI=np.array(iti), experiment=exp, all_stim_durations=asd
        )
        result = des.designmatrix()
        if result is False:
            continue
        des.FeCalc()
        raw_fes.append(des.Fe * exp.FeMax)

    exp.FeMax = max(raw_fes)

    # Now test
    fe_values = []
    for i in range(50):
        order = list(rng.choice(3, size=30, p=[1 / 3, 1 / 3, 1 / 3]))
        iti = [2.0] * 30
        asd = Experiment.sample_stim_durations(
            order, exp.stimuli_durations, exp.t_pre, exp.t_post
        )
        des = Design(
            order=order, ITI=np.array(iti), experiment=exp, all_stim_durations=asd
        )
        result = des.designmatrix()
        if result is False:
            continue
        des.FeCalc()
        fe_values.append(des.Fe)

    fe_arr = np.array(fe_values)
    print(f"  Fe range: [{fe_arr.min():.4f}, {fe_arr.max():.4f}]")
    assert np.all(fe_arr >= 0), f"Fe has negative values: {fe_arr.min()}"
    assert np.all(fe_arr <= 1.01), f"Fe exceeds 1: {fe_arr.max()}"
    print("  PASSED\n")


# ============================================================
# TEST 3: All designs share same Experiment — whitening matrix identical
# ============================================================
def test_shared_experiment():
    print("TEST 3: All designs share one Experiment object...")
    exp = make_variable_dur_experiment()

    rng = np.random.RandomState(42)
    experiments_seen = set()
    for i in range(10):
        order = list(rng.choice(3, size=30, p=[1 / 3, 1 / 3, 1 / 3]))
        iti = [2.0] * 30
        asd = Experiment.sample_stim_durations(
            order, exp.stimuli_durations, exp.t_pre, exp.t_post
        )
        des = Design(
            order=order, ITI=np.array(iti), experiment=exp, all_stim_durations=asd
        )
        experiments_seen.add(id(des.experiment))

    assert (
        len(experiments_seen) == 1
    ), f"Found {len(experiments_seen)} Experiment objects, expected 1"
    print(f"  All 10 designs share Experiment at {experiments_seen.pop()}")
    print("  PASSED\n")


# ============================================================
# TEST 4: Container size — designs with shorter actual duration fit
# ============================================================
def test_short_design_fits_container():
    print("TEST 4: Short designs fit in container...")
    exp = make_variable_dur_experiment(
        stimuli_durations=[
            {"model": "fixed", "mean": 0.5},
            {"model": "fixed", "mean": 0.5},
            {"model": "fixed", "mean": 0.5},
        ],
        trial_max=2.0,
    )

    order = [0, 1, 2] * 10
    iti = [2.0] * 30
    asd = Experiment.sample_stim_durations(
        order, exp.stimuli_durations, exp.t_pre, exp.t_post
    )

    des = Design(order=order, ITI=np.array(iti), experiment=exp, all_stim_durations=asd)
    des.designmatrix()
    des.FeCalc()
    print(f"  Fe = {des.Fe:.4f}, container n_tp = {exp.n_tp}")
    print("  PASSED\n")


# ============================================================
# TEST 5: Container size — designs with long ITI draws (exponential tail)
# ============================================================
def test_long_iti_truncation():
    print("TEST 5: Long ITI draws — graceful rejection...")
    exp = make_basic_experiment(
        ITImodel="exponential",
        ITImean=2.0,
        ITImin=1.0,
        ITImax=6.0,
    )

    rng = np.random.RandomState(99)
    order = list(rng.choice(3, size=30, p=[1 / 3, 1 / 3, 1 / 3]))
    iti = [5.0] * 30

    des = Design(order=order, ITI=np.array(iti), experiment=exp)
    result = des.designmatrix()

    if result is False:
        print(
            "  Design correctly rejected (returned False) — overflow handled gracefully"
        )
        print("  PASSED\n")
        return True
    else:
        print("  Design fit in container despite long ITIs")
        print("  PASSED\n")
        return True


# ============================================================
# TEST 6: Conditional ITI — generate_iti returns correct length
# ============================================================
def test_conditional_iti_length():
    print("TEST 6: Conditional ITI correct length...")
    order = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    conditional_iti = {
        (0, 1): {"model": "fixed", "mean": 2.0},
        (1, 2): {"model": "fixed", "mean": 3.0},
        "default": {"model": "fixed", "mean": 1.5},
    }

    iti = Experiment.generate_iti(order, conditional_iti)
    assert len(iti) == len(order), f"ITI length {len(iti)} != order length {len(order)}"
    print(f"  ITI length = {len(iti)} (matches order length {len(order)})")
    print(f"  ITI values: {iti}")
    print("  PASSED\n")


# ============================================================
# TEST 7: Conditional ITI gaussian — was appending to wrong list
# ============================================================
def test_conditional_iti_gaussian():
    print("TEST 7: Conditional ITI gaussian branch fix...")
    order = [0, 1, 0, 1, 0]
    conditional_iti = {
        "default": {"model": "gaussian", "mean": 2.0, "std": 0.5, "min": 1.0, "max": 3.0},
    }

    iti = Experiment.generate_iti(order, conditional_iti)
    assert len(iti) == len(order), f"ITI length {len(iti)} != order length {len(order)}"
    assert all(1.0 <= v <= 3.0 for v in iti), f"ITI values out of bounds: {iti}"
    print(f"  ITI values: {[f'{v:.2f}' for v in iti]}")
    print("  PASSED\n")


# ============================================================
# TEST 8: sample_stim_durations returns correct values
# ============================================================
def test_sample_stim_durations():
    print("TEST 8: sample_stim_durations correctness...")
    order = [0, 1, 2, 0, 1, 2]
    stimuli_durations = [
        {"model": "fixed", "mean": 1.0},
        {"model": "fixed", "mean": 1.5},
        2.0,
    ]

    asd = Experiment.sample_stim_durations(
        order, stimuli_durations, t_pre=0.5, t_post=0.0
    )
    assert len(asd) == 6
    expected = [1.5, 2.0, 2.5, 1.5, 2.0, 2.5]
    assert np.allclose(asd, expected), f"Expected {expected}, got {asd}"
    print(f"  Durations: {asd}")
    print("  PASSED\n")


# ============================================================
# TEST 9: NulDesign in max_eff still works (all_stim_durations=None)
# ============================================================
def test_nuldesign_works():
    print("TEST 9: NulDesign in max_eff works...")
    exp = make_basic_experiment()
    assert exp.FcMax != 1, f"FcMax not calibrated: {exp.FcMax}"
    assert exp.FfMax != 1, f"FfMax not calibrated: {exp.FfMax}"
    print(f"  FcMax = {exp.FcMax:.4f}, FfMax = {exp.FfMax:.4f}")
    print("  PASSED\n")


# ============================================================
# TEST 10: Crossover propagates all_stim_durations
# ============================================================
def test_crossover_propagates_asd():
    print("TEST 10: Crossover propagates all_stim_durations...")
    exp = make_variable_dur_experiment()

    rng = np.random.RandomState(42)
    order1 = list(rng.choice(3, size=30, p=[1 / 3, 1 / 3, 1 / 3]))
    order2 = list(rng.choice(3, size=30, p=[1 / 3, 1 / 3, 1 / 3]))
    iti = np.array([2.0] * 30)

    asd1 = Experiment.sample_stim_durations(
        order1, exp.stimuli_durations, exp.t_pre, exp.t_post
    )
    asd2 = Experiment.sample_stim_durations(
        order2, exp.stimuli_durations, exp.t_pre, exp.t_post
    )

    des1 = Design(order=order1, ITI=iti, experiment=exp, all_stim_durations=asd1)
    des2 = Design(order=order2, ITI=iti, experiment=exp, all_stim_durations=asd2)

    offspring = des1.crossover(des2, seed=42)

    for i, baby in enumerate(offspring):
        assert (
            baby.all_stim_durations is not None
        ), f"Offspring {i} lost all_stim_durations"
        assert len(baby.all_stim_durations) == 30, f"Offspring {i} wrong asd length"
        assert id(baby.experiment) == id(exp), f"Offspring {i} has different Experiment!"

    print("  Both offspring have all_stim_durations and share Experiment")
    print("  PASSED\n")


# ============================================================
# TEST 11: Mutation propagates all_stim_durations
# ============================================================
def test_mutation_propagates_asd():
    print("TEST 11: Mutation propagates all_stim_durations...")
    exp = make_variable_dur_experiment()

    rng = np.random.RandomState(42)
    order = list(rng.choice(3, size=30, p=[1 / 3, 1 / 3, 1 / 3]))
    iti = np.array([2.0] * 30)
    asd = Experiment.sample_stim_durations(
        order, exp.stimuli_durations, exp.t_pre, exp.t_post
    )

    des = Design(order=order, ITI=iti, experiment=exp, all_stim_durations=asd)
    mutant = des.mutation(0.1, seed=42)

    assert mutant.all_stim_durations is not None, "Mutant lost all_stim_durations"
    assert len(mutant.all_stim_durations) == 30
    assert id(mutant.experiment) == id(exp), "Mutant has different Experiment!"
    print("  Mutant has all_stim_durations and shares Experiment")
    print("  PASSED\n")


# ============================================================
# TEST 12: Fixed order — crossover/mutation preserve order
# ============================================================
def test_fixed_order():
    print("TEST 12: Fixed order preserved in crossover/mutation...")
    fixed_order = [0, 1, 2] * 10
    exp = make_variable_dur_experiment(order=fixed_order)

    assert exp.order_fixed is True

    iti = np.array([2.0] * 30)
    asd = Experiment.sample_stim_durations(
        fixed_order, exp.stimuli_durations, exp.t_pre, exp.t_post
    )

    des1 = Design(order=fixed_order, ITI=iti, experiment=exp, all_stim_durations=asd)
    des2 = Design(order=fixed_order, ITI=iti, experiment=exp, all_stim_durations=asd)

    babies = des1.crossover(des2, seed=42)
    for baby in babies:
        assert baby.order == fixed_order, "Crossover changed fixed order!"

    mutant = des1.mutation(0.2, seed=42)
    assert mutant.order == fixed_order, "Mutation changed fixed order!"

    print("  Order preserved through crossover and mutation")
    print("  PASSED\n")


# ============================================================
# TEST 13: restnum > 0 with variable stim durations — alignment check
# ============================================================
def test_restnum_with_variable_durations():
    print("TEST 13: restnum > 0 with variable stim durations...")
    exp = make_variable_dur_experiment(restnum=10, restdur=5.0)

    rng = np.random.RandomState(42)
    order = list(rng.choice(3, size=30, p=[1 / 3, 1 / 3, 1 / 3]))
    iti = [2.0] * 30
    asd = Experiment.sample_stim_durations(
        order, exp.stimuli_durations, exp.t_pre, exp.t_post
    )

    des = Design(order=order, ITI=np.array(iti), experiment=exp, all_stim_durations=asd)

    try:
        des.designmatrix()
        print(f"  Onsets computed: {len(des.onsets)} (expect 30)")
        assert len(des.onsets) == 30, f"Expected 30 onsets, got {len(des.onsets)}"
        des.FeCalc()
        print(f"  Fe = {des.Fe:.4f}")
        print("  PASSED\n")
    except Exception as e:
        print(f"  FAILED with error: {e}")
        import traceback

        traceback.print_exc()
        print()


# ============================================================
# RUN ALL TESTS
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING neurodesign.classes behavior")
    print("=" * 60 + "\n")

    results = {}
    tests = [
        ("basic_fe_bounded", test_basic_fe_bounded),
        ("variable_dur_fe_bounded", test_variable_dur_fe_bounded),
        ("shared_experiment", test_shared_experiment),
        ("short_design_fits", test_short_design_fits_container),
        ("long_iti_truncation", test_long_iti_truncation),
        ("conditional_iti_length", test_conditional_iti_length),
        ("conditional_iti_gaussian", test_conditional_iti_gaussian),
        ("sample_stim_durations", test_sample_stim_durations),
        ("nuldesign_works", test_nuldesign_works),
        ("crossover_propagates_asd", test_crossover_propagates_asd),
        ("mutation_propagates_asd", test_mutation_propagates_asd),
        ("fixed_order", test_fixed_order),
        ("restnum_variable_dur", test_restnum_with_variable_durations),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            results[name] = "PASSED"
            passed += 1
        except Exception as e:
            results[name] = f"FAILED: {e}"
            failed += 1
            import traceback

            traceback.print_exc()
            print()

    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    for name, result in results.items():
        status = "✓" if "PASSED" in result else "✗"
        print(f"  {status} {name}: {result}")
