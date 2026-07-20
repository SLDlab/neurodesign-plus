from neurodesign import Optimisation, report


def test_report_smoke(case10_experiment, tmp_path):
    pop = Optimisation(
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
    pop.optimise()
    pop.download()
    report.make_report(pop, tmp_path / "test.pdf")
    assert (tmp_path / "test.pdf").exists()
