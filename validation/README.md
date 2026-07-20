# Validation

This directory contains maintained validation code, small stable fixtures consumed by tests, and manuscript-support generators. Generated validation outputs are written to `validation/_artifacts/`, which is intentionally ignored.

Maintained source:

- `manifest.py`: authoritative workflow inventory for maintained release-validation entrypoints
- `run_all_validation_workflows.py`: aggregate validation runner
- `execute_notebooks.py`: executes the tracked release-critical notebooks into an output copy tree
- `compare_case10_v1_vs_v2.py`: canonical version-1 versus version-2 Case 10 comparison
- `helpers/`: shared canonical Case 10 specification and version metadata helpers
- `manuscript_support/`: manuscript-facing figure generators and wrappers

Tracked stable fixtures:

- version-1 reference metadata lives under `tests/fixtures/v1_reference/`
- each fixture there includes a README documenting provenance and consuming tests

Generated outputs:

- default output root: `validation/_artifacts/`
- the aggregate runner writes `validation_inventory.json` and `validation_results.json` there
- notebook execution copies, docs builds, Case 10 comparisons, and manuscript-support bundles are also written there

How to run all maintained validations:

```bash
python -m validation.run_all_validation_workflows
```

How failures are reported:

- each workflow records stdout and stderr logs
- `validation/_artifacts/run_all/validation_results.json` stores per-workflow status, runtime, command, and failure summary (the `run_all` artifact-directory name predates the script rename and is left as-is since it's a generated output path, not source code)
- a nonzero exit code means at least one maintained workflow failed
