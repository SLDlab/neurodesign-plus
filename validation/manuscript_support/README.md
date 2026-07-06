# Manuscript Support

This directory stores the current accepted validation-side provenance bundle for manuscript-facing figures.

Version 2.0 migration note:

- `generate_tutorial3_case10_v2_package.py` is the version-2 timing-architecture runner.
- `generate_tutorial3_case10_nontrivial_package.py` is now a thin compatibility wrapper around the version-2 runner so the manuscript-support entrypoint name remains stable while the generated content uses the canonical v2 Case 10 specification.

Contents:

- `architecture_schematic.svg`: version-2 timing-architecture schematic
- `case10_nontrivial_design_figure.png`: schedule-style timing figure for the first four complete conceptual trials of the selected design
- `case10_nontrivial_convolved_design_and_predicted_bold.png`: combined convolved design matrix and predicted BOLD export for the selected design
- `report_page-1.png` through `report_page-4.png`: rendered report pages from the package-generated PDF report
- `figure_provenance.json`: provenance and caption-support metadata
- `artifacts_v2/case10_v2_metadata.json`: machine-readable run record including the selected design export and optimisation trace
- `generate_tutorial3_case10_nontrivial_package.py`: stable wrapper entrypoint
- `generate_tutorial3_case10_v2_package.py`: canonical version-2 manuscript-support generator

Source and scope:

- case identifier: `case_10_combined_branching_design_nontrivial_template_optimisation_v2`
- case mode: `canonical version-2 Tutorial 3 Case 10 using probabilistic conceptual templates inside Optimisation`
- status: current version-2 manuscript-support bundle as of `2026-07-05`

Cleanup status:

- this directory is intentionally flattened; the accepted manuscript-support files now live directly in `validation/manuscript_support/`
- superseded support bundles `manuscript_figure_package_tutorial3_case10_2026-07-04/` and `manuscript_figure_package_2026-07-04/` were deleted during cleanup

The manuscript-facing image files are mirrored into `manuscript/docs_images/`. This directory retains the supporting provenance, generation code, and bounded historical context.
