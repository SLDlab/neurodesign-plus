# Installation & Setup

This `neurodesign-plus` repository provides an extended and maintained fork of the `neurodesign` package.
These instructions are for working with the source code locally (development or research use).

---

## I. Clone the repository

Clone the `neurodesign-plus` repository and move into it:

```bash
git clone https://github.com/SLDlab/neurodesign-plus.git
cd neurodesign-plus
```

---

## II. Create a virtual environment

Create a Python virtual environment from the repository root:

You can do this either through the VS Code interface or directly in the terminal.

**A. Using VS Code GUI:**

1. Open the Command Palette:
   `View > Command Palette`
2. Search for and select:
   `Python: Select Interpreter`
3. Click:
   `+ Create Environment`
4. Choose the environment type:
   `venv`
5. Select a Python interpreter (version `>= 3.9`, e.g., `Python 3.13.1`)

**B. Using the Terminal:**

From the root of the project folder, run:

```bash
python -m venv venv
```

---

## III. Activate the virtual environment

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

---

## IV. Install dependencies and the package (editable mode)

With the virtual environment activated, install the dependencies:

```bash
pip install -r requirements.txt
pip install .
```

To deactivate the environment when you're done:

```bash
deactivate
```

---

## V. (Optional) Use the environment in Jupyter notebooks

If you're working in `.ipynb` notebooks, register the virtual environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name=venv --display-name "Python (neurodesign venv)"
```

Reload VSC. Then, in Jupyter, select the `"Python (neurodesign venv)"` kernel when working on notebooks.

---

## Notes

- This fork replaces the upstream `neurodesign` package.
- Do not install the upstream repository separately.
