# Running a Jupyter Notebook from GitHub Without an Existing Codespace

## Overview

This guide explains how to run a Jupyter Notebook stored in a GitHub repository when no GitHub Codespace has been created yet. The process includes:
- Accessing the GitHub repository
- Creating a Codespace
- Installing required dependencies
- Launching Jupyter Notebook
- Running notebook cells

This guide assumes:
- You have a GitHub account
- The repository already exists on GitHub
- The notebook file (`.ipynb`) is already present in the repository

---

# Requirements

Before starting, ensure you have:
- A GitHub account
- Internet access
- Permission to access the repository

Optional but recommended:
- Familiarity with basic terminal commands
- A modern web browser

---

# Opening the GitHub Repository

1. Open your web browser.
2. Navigate to:

```text
https://github.com
```

3. Sign in to your GitHub account.
4. Open the repository containing the Jupyter Notebook.

Example repository URL:

```text
https://github.com/username/repository-name
```

---

# Creating a Codespace

Since no Codespace exists yet, one must be created.

## Step 1 — Open the Repository

Inside the repository page:
1. Locate the green button labeled:

```text
Code
```

2. Click the button.

---

## Step 2 — Open the Codespaces Tab

Inside the menu:
1. Select the tab:

```text
Codespaces
```

---

## Step 3 — Create a New Codespace

1. Click:

```text
Create codespace on main
```

If the repository uses a different branch name, the button may instead display:
- `Create codespace on master`
- `Create codespace on dev`
- or another branch name

GitHub will begin creating the development environment.

This process may take several minutes.

---

# Waiting for Codespace Initialization

When the Codespace finishes loading:
- A VS Code-style browser interface will appear
- The repository files will be visible in the left file explorer
- A terminal may open automatically

If the terminal does not appear:
1. Select:

```text
Terminal → New Terminal
```

---

# Locating the Jupyter Notebook

In the left file explorer:
1. Navigate through the repository folders
2. Locate the `.ipynb` notebook file

Example:

```text
solar_sizing.ipynb
```

3. Click the notebook file to open it

The notebook editor will open inside the browser.

---

# Installing Required Python Packages

Before running the notebook, install the required Python packages.

In the terminal, run:

```bash
pip install pvlib plotly notebook ipython pandas numpy matplotlib scipy requests pytz
```

If the repository contains a `requirements.txt` file, install dependencies using:

```bash
pip install -r requirements.txt
```

If the repository contains an `environment.yml` file, use:

```bash
conda env create -f environment.yml
```

and activate the environment using:

```bash
conda activate <environment_name>
```

Replace `<environment_name>` with the actual environment name defined in the file.

---

# Verifying Package Installation

To verify installation, run:

```bash
pip list
```

Check that required packages appear in the installed package list.

Example packages:
- `pvlib`
- `plotly`
- `pandas`
- `numpy`
- `matplotlib`

---

# Selecting the Python Kernel

Inside the notebook:
1. Locate the kernel selector in the upper-right corner
2. Select the correct Python environment

Typical kernel name:

```text
Python 3
```

If prompted:
1. Select:

```text
Install/Enable Suggested Extensions
```

2. Wait for installation to complete

---

# Running Notebook Cells

## Running a Single Cell

To run a single notebook cell:
1. Click inside the cell
2. Press:

```text
Shift + Enter
```

The cell will execute and move to the next cell.

---

## Running All Cells

To run the entire notebook:
1. Select:

```text
Run → Run All Cells
```

or:

```text
Kernel → Restart Kernel and Run All Cells
```

Cells must execute sequentially from top to bottom.

---

# Editing Notebook Parameters

Many notebooks contain configuration cells near the beginning of the notebook.

Typical editable parameters include:
- Coordinates
- File paths
- Simulation settings
- Financial parameters
- PV installation parameters

Example:

```python
latitude = 55.734
longitude = 13.248
surface_tilt = 41
```

Update these values before running simulations.

---

# Viewing Outputs

Notebook outputs appear directly below each cell.

Typical outputs include:
- Tables
- Figures
- Plotly interactive graphs
- Financial analysis
- Energy simulations

Interactive plots may require a few seconds to load.

---

# Stopping the Codespace

When finished:
1. Return to the GitHub repository page
2. Open the:

```text
Codespaces
```

tab
3. Locate the active Codespace
4. Select:

```text
Stop Codespace
```

Stopping unused Codespaces helps avoid unnecessary resource usage.

---

# Troubleshooting

## Codespace Does Not Start

Possible causes:
- GitHub service interruption
- Browser issue
- Repository permission issue

Solutions:
- Refresh the browser
- Retry Codespace creation
- Verify repository access permissions

---

## Notebook Does Not Open

Verify:
- The file extension is `.ipynb`
- The notebook exists in the repository
- The Codespace finished loading completely

---

## Python Packages Missing

Install packages manually:

```bash
pip install <package_name>
```

Example:

```bash
pip install pvlib
```

---

## Kernel Not Available

Install the Python extension if prompted and reload the Codespace.

---

## Plotly Figures Do Not Display

Update Plotly:

```bash
pip install --upgrade plotly
```

---

# Recommended Workflow

## Step 1 — Open the GitHub repository

Open the repository in your browser.

---

## Step 2 — Create a Codespace

Use:

```text
Code → Codespaces → Create codespace on main
```

---

## Step 3 — Wait for initialization

Allow the Codespace environment to finish loading.

---

## Step 4 — Install dependencies

Run:

```bash
pip install -r requirements.txt
```

or install packages manually.

---

## Step 5 — Open the notebook

Open the `.ipynb` file from the file explorer.

---

## Step 6 — Select the Python kernel

Choose the correct Python environment.

---

## Step 7 — Edit notebook parameters

Update configuration values as needed.

---

## Step 8 — Run notebook cells

Use:

```text
Shift + Enter
```

or run all notebook cells.

---

## Step 9 — Review outputs

Check generated plots, tables, and calculations.

---