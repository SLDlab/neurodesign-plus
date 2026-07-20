# Version-1 Reference Fixtures

This directory contains the minimal tracked Neurodesign-Plus 1.x reference data
still required by the maintained version-2 regression tests.

- `case10_tutorial_metadata.json`
  - produced by the archived version-1 Tutorial 3 Case 10 baseline capture
  - consumed by `tests/test_classes.py::test_case10_tutorial_and_manuscript_support_share_spec`
  - purpose: preserve the legacy Case 10 tutorial template and conditional-timing reference
  - type: numerical/provenance fixture

- `case10_manuscript_support_metadata.json`
  - produced by the archived version-1 manuscript-support Case 10 baseline capture
  - consumed by `tests/test_classes.py::test_case10_tutorial_and_manuscript_support_share_spec`
  - purpose: preserve the legacy Case 10 manuscript-support template and conditional-timing reference
  - type: numerical/provenance fixture

These files were extracted from the dated local Phase 1 baseline archive so the
tests no longer depend on `validation/phase1_baseline_2026-07-05/`.
