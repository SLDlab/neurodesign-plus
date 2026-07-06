# Installation And Setup

`neurodesign-plus` is the install distribution.
`neurodesign` is the Python import package.

The project uses `pyproject.toml` as the maintained installation surface. These instructions use `.venv` consistently and avoid a stale `requirements.txt` workflow.

## Create And Activate A Virtual Environment

From the repository root:

```bash
python -m venv .venv
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

## End-User Installation From PyPI

Install the published package and its runtime dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install neurodesign-plus
```

Use it as:

```python
from neurodesign import Experiment, Design, Optimisation
```

## Local Source Installation

Install the current checkout as a standard local package:

```bash
python -m pip install .
```

This is the right choice when you want the local code without editable development tooling.

## Editable Development Installation

Install the editable package plus all declared development extras:

```bash
python -m pip install -e ".[dev]"
```

The `dev` extra includes the declared `doc` and `test` extras plus `pre-commit` and `tox`.

## Test And Notebook Dependencies

If you only need the package plus test and notebook execution dependencies:

```bash
python -m pip install -e ".[test]"
```

The `test` extra installs `pytest`, coverage tools, and notebook execution dependencies such as `nbclient`, `nbformat`, and `nbmake`.

## Documentation Dependencies

If you only need the package plus documentation build dependencies:

```bash
python -m pip install -e ".[doc]"
```

The `doc` extra installs the maintained Sphinx toolchain declared in `pyproject.toml`.

## Report Dependencies

No extra is currently required for report generation.
The base runtime install already includes the report stack used by version 2.0: `matplotlib`, `reportlab`, and `pdfrw`.

## Notebook Kernel Registration

To expose the local environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name neurodesign-plus --display-name "Python (neurodesign-plus)"
```

## Verification

Quick import check:

```bash
python -c "from neurodesign import Experiment, Design, Optimisation; print(Experiment.__module__)"
```

The import should resolve from `neurodesign`, not from a separate upstream package.
