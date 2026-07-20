import numpy as np

from neurodesign import Experiment


def test_flat_generated_design_smoke():
    exp = Experiment(
        TR=2.0,
        n_trials=12,
        P=[0.25, 0.25, 0.25, 0.25],
        C=[[1, 0, 0, -1]],
        n_stimuli=4,
        rho=0.3,
        event_durations=1.0,
        trial_start_interval=0.0,
        post_event_interval=0.0,
        inter_trial_interval={
            "model": "exponential",
            "mean": 2.0,
            "min": 1.0,
            "max": 4.0,
        },
        resolution=0.1,
        seed=99,
    )
    design = exp.create_design(seed=99)
    assert len(design.order) == 12
    assert np.all(design.realized_inter_trial_intervals >= 1.0)
    assert np.all(design.realized_inter_trial_intervals <= 4.0)
