# Manuscript Support

This directory stores the current accepted validation-side provenance bundle for manuscript-facing figures.

Contents:

- `architecture_schematic.svg` / `architecture_schematic.png`: version-2 timing-architecture schematic
- `case10_nontrivial_design_figure.png`: schedule-style timing figure for the first four complete conceptual trials of the selected design
- `case10_nontrivial_convolved_design_and_predicted_bold.png`: combined convolved design matrix and predicted BOLD export for the selected design
- `case10_nontrivial_report_page-1.png` through `case10_nontrivial_report_page-4.png`: rendered pages from the package-generated PDF report for this Case 10 run (rasterized with `pymupdf`; prefixed with `case10_nontrivial_` so they don't collide with any other notebook's own report-page renders, e.g. `tutorial_base-optimizing_and_reporting.ipynb`'s `base_optimizing_reporting_report_page-*.png`)
- `figure_provenance.json`: provenance and caption-support metadata
- `artifacts_v2/case10_v2_metadata.json`: machine-readable run record including the selected design export and optimisation trace
- `generate_case10_manuscript_figures.py`: canonical manuscript-support generator (renamed 2026-07-16 from `generate_tutorial3_case10_v2_package.py`; the redundant `generate_tutorial3_case10_nontrivial_package.py` compatibility wrapper was deleted in the same pass since nothing in the repo called it). Also renders the report-page PNGs directly (a prior version of this script lost that step when it was renamed; it's back as `_render_report_pages()`).

Source and scope:

- case identifier: `case_10_combined_branching_design_nontrivial_template_optimisation_v2`
- case mode: `canonical version-2 Tutorial 3 Case 10 using probabilistic conceptual templates inside Optimisation`
- status: current version-2 manuscript-support bundle as of `2026-07-05`

Cleanup status:

- this directory is intentionally flattened; the accepted manuscript-support files now live directly in `validation/manuscript_support/`
- superseded support bundles `manuscript_figure_package_tutorial3_case10_2026-07-04/` and `manuscript_figure_package_2026-07-04/` were deleted during cleanup

The manuscript-facing image files are mirrored into both `manuscript/docs_images/` (manuscript build) and `docs/_images/` (Sphinx/ReadTheDocs site), via `DOCS_IMAGES_DIRS` in each generator script. This directory retains the supporting provenance, generation code, and bounded historical context.
