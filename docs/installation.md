# Installation

## From PyPI (recommended)

```bash
pip install neurodesign-plus
```

This installs the package and all required dependencies. You can then use it as:

```python
from neurodesign import Experiment, Design, Optimisation
```

:::{note}
The PyPI package is called `neurodesign-plus`, but the Python import remains `neurodesign`.
This fork replaces the upstream `neurodesign` package -- do **not** install both.
:::

## From Source (development)

For development or to access the latest unreleased changes:

### 1. Clone the repository

```bash
git clone https://github.com/SLDlab/neurodesign-plus.git
cd neurodesign-plus
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install in editable mode

```bash
pip install -e ".[dev]"
```

This installs the package in editable mode along with all development dependencies (testing, linting, documentation).

### 5. (Optional) Jupyter kernel

If you are working with the tutorial notebooks, register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name=neurodesign-plus --display-name "Python (neurodesign-plus)"
```
