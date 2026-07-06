# Installation

`neurodesign-plus` is the install distribution.
`neurodesign` is the import package.

## End-User Installation From PyPI

```bash
python -m pip install --upgrade pip
python -m pip install neurodesign-plus
```

Use it as:

```python
from neurodesign import Experiment, Design, Optimisation
```

The base install already includes report-generation dependencies.
No separate report extra exists in `pyproject.toml`.

## Local Source Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/SLDlab/neurodesign-plus.git
cd neurodesign-plus
python -m venv .venv
```

Activate the environment:

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the local checkout:

```bash
python -m pip install .
```

## Editable Development Installation

Install the editable package plus the declared development extras:

```bash
python -m pip install -e ".[dev]"
```

The `dev` extra includes the maintained `doc` and `test` extras plus `pre-commit` and `tox`.

## Test And Notebook Dependencies

Install the package plus the declared testing and notebook execution dependencies:

```bash
python -m pip install -e ".[test]"
```

## Documentation Dependencies

Install the package plus the maintained Sphinx toolchain:

```bash
python -m pip install -e ".[doc]"
```

## Version-2 Migration

Version 2.0 is a clean breaking API release.
Use the [migration guide](migration.md) for version-1 name mapping and workflow changes instead of trying to preserve removed timing aliases in-place.
