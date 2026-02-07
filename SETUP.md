# Installation & Setup

To use this modified version, clone this repository and install the requirements.

### Setting Up `neurodesign` and `deconvolve`

To work with the `neurodesign` and `deconvolve` packages for experiment creation, we recommend setting up a Python virtual environment. This ensures that your project dependencies remain isolated and reproducible.

---

### I. Clone and open the repository folder in Visual Studio Code.

---

### II. Create a virtual environment

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

### III. Activate the virtual environment

- On macOS/Linux:
  ```bash
  source .venv/bin/activate
  ```
- On Windows:
  ```bash
  venv\Scripts\Activate.ps1
  ```

---

### IV. Install required packages

With the virtual environment activated, install dependencies:
```bash
pip install git+https://github.com/neuropower/neurodesign.git
pip install -r requirements.txt
```

To deactivate the environment when you're done:
```bash
deactivate
```
---

### V. Applying the modified classes.py

Within your virtual environment, you will need to replace the original toolbox's classes.py with the modified classes-1.py.

Here is an **example filepath** of where the classes.py can be found based on your python installation.

```bash
./.venv/lib/python3.13/site-packages/neurodesign/classes.py
```

Remove this classes.py file and replace it with the classes-1.py renamed as classes.py.

Once this modification has been done in your environment, the tool is ready for use.

---

### VI. (Optional) Use the virtual environment in Jupyter Notebooks

If you're working in `.ipynb` notebooks, register the virtual environment as a Jupyter kernel:
```bash
python -m ipykernel install --user --name=venv --display-name "Python (neurodesign venv)"
```

Reload VSC. Then, in Jupyter, select the `"Python (neurodesign venv)"` kernel when working on notebooks.

---

**Note:** At this stage, we are primarily using the `neurodesign` package.
