from validation.manifest import WORKFLOWS


def test_validation_manifest_contains_release_critical_workflows():
    names = {workflow["name"] for workflow in WORKFLOWS}
    assert {
        "pytest",
        "docs",
        "notebooks",
        "case10_comparison",
        "manuscript_case10",
        "case10_determinism",
        "timing_svg",
    }.issubset(names)


def test_validation_manifest_names_are_unique():
    names = [workflow["name"] for workflow in WORKFLOWS]
    assert len(names) == len(set(names))
