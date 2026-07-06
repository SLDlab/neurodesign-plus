from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

WORKFLOWS = [
    {
        "name": "pytest",
        "kind": "maintained release validation",
        "command": [
            "-m",
            "pytest",
            "tests",
            "-q",
            "--junitxml",
            "{artifact_dir}/pytest_junit.xml",
        ],
        "assertion_summary": "Runs the full automated test suite, including timing, exports, optimization, and compatibility checks.",
        "artifact_subdir": "pytest",
        "ci": True,
    },
    {
        "name": "docs",
        "kind": "maintained release validation",
        "command": [
            "-m",
            "sphinx",
            "-W",
            "--keep-going",
            "-b",
            "html",
            "docs",
            "{artifact_dir}/html",
        ],
        "assertion_summary": "Builds the documentation with warnings treated as errors.",
        "artifact_subdir": "docs",
        "ci": True,
    },
    {
        "name": "notebooks",
        "kind": "maintained release validation",
        "command": [
            "-m",
            "validation.execute_notebooks",
            "--output-dir",
            "{artifact_dir}",
        ],
        "assertion_summary": "Executes every release-critical notebook from a clean kernel without modifying tracked notebook files.",
        "artifact_subdir": "notebooks",
        "ci": True,
    },
    {
        "name": "case10_comparison",
        "kind": "maintained release validation",
        "command": ["-m", "validation.run_case10_v2"],
        "assertion_summary": "Builds the canonical Case 10 tutorial and manuscript-support comparison against stable version-1 fixtures.",
        "artifact_subdir": "case10_comparison",
        "ci": False,
    },
    {
        "name": "manuscript_case10",
        "kind": "manuscript-support generator",
        "command": [
            "-m",
            "validation.manuscript_support.generate_tutorial3_case10_v2_package",
        ],
        "assertion_summary": "Regenerates the canonical manuscript-support Case 10 package, report, exports, and figures.",
        "artifact_subdir": "manuscript_support",
        "ci": False,
    },
    {
        "name": "case10_determinism",
        "kind": "maintained release validation",
        "command": ["-m", "validation.run_case10_determinism"],
        "assertion_summary": "Runs the canonical Case 10 optimisation twice in independent processes and compares the full selected-design payloads.",
        "artifact_subdir": "case10_determinism",
        "ci": False,
    },
    {
        "name": "timing_svg",
        "kind": "maintained example generator",
        "command": [
            "-m",
            "validation.manuscript_support.generate_timing_architecture_svg",
        ],
        "assertion_summary": "Regenerates the timing-architecture SVG and rendered preview.",
        "artifact_subdir": "timing_svg",
        "ci": False,
    },
]
