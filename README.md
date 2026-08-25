# ANCHOR

**Automated Non-linear Curve fitting for High-speed Observed Reactions**

ANCHOR is a Python-based graphical application for automated fitting and model selection of time-resolved kinetic data. It is designed for stopped-flow and related progress-curve experiments and can evaluate sequential kinetic mechanisms, direct turnover, reversible Michaelis-Menten turnover, linear behavior, and no-binding behavior within a single workflow.

ANCHOR combines bounded multistart optimization, numerical ODE integration, robust fitting, AIC/BIC-based model selection, and noise-aware no-binding safeguards. The program produces trace-level parameter tables, diagnostic plots, model-selection summaries, species profiles, and optional Michaelis-Menten analyses.

## Features

- Graphical user interface built with Tkinter
- Automated model selection for time-resolved kinetic traces
- Sequential kinetic models containing 0–5 intermediates
- Slow-engagement kinetic models
- Direct turnover model:

  \[
  E + S \rightarrow P
  \]

- Reversible Michaelis-Menten turnover model:

  \[
  E + S \rightleftharpoons ES \rightarrow P
  \]

- Linear-only and no-binding diagnostic models
- Bounded multistart nonlinear optimization
- Robust soft-L1 loss
- Numerical ODE integration
- AIC/BIC-based model ranking
- Noise-aware no-binding guardrail
- Optional Michaelis-Menten post-fit analysis
- Per-trace, species, concentration-specific, and combined plots
- CSV, JSON, Excel, and PNG outputs
- Reproducible configuration files
- Example datasets and preconfigured example setups

## Repository contents

| File or folder | Description |
| --- | --- |
| `ANCHOR.py` | Main ANCHOR application |
| `ANCHOR_requirements.txt` | Python dependencies |
| `ANCHOR_Quick_Start.txt` | Short installation and startup guide |
| `ANCHOR_User_Guide.pdf` | Full user guide |
| `ANCHOR_Example_Testing_Guide.pdf` | Detailed instructions for the included validation examples |
| `ANCHOR_Example_Testing_Quick_Checklist.txt` | Quick reference for testing the example datasets |
| `examples/` | Individual example input datasets |
| `example_inputs.zip` | Packaged copy of the example inputs |
| `setup_configs/` | Preconfigured JSON setup files for the four examples |

## Installation

### 1. Download ANCHOR

Clone the repository:

```bash
git clone <repository-url>
cd ANCHOR
```

Alternatively, download the repository as a ZIP file from GitHub and extract it.

### 2. Create a Python virtual environment

A virtual environment is recommended so that ANCHOR's dependencies remain isolated from other Python projects.

**Windows**

```bash
python -m venv anchor_env
anchor_env\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv anchor_env
source anchor_env/bin/activate
```

### 3. Install the required Python packages

Using the included requirements file:

```bash
python -m pip install -r ANCHOR_requirements.txt
```

or install the packages directly:

```bash
python -m pip install numpy pandas matplotlib scipy openpyxl
```

ANCHOR requires:

- NumPy
- pandas
- Matplotlib
- SciPy
- openpyxl

### 4. Verify Tkinter

ANCHOR uses Tkinter for its graphical interface.

```bash
python -m tkinter
```

A small Tkinter test window should appear.

Tkinter is normally included with standard Python installations on Windows and macOS. Some Linux distributions may require installation of the system Tkinter package, for example:

```bash
sudo apt install python3-tk
```

## Running ANCHOR

From the repository directory:

```bash
python ANCHOR.py
```

On systems where Python 3 is invoked as `python3`:

```bash
python3 ANCHOR.py
```

The ANCHOR graphical interface will open.

## Input data format

ANCHOR accepts a CSV file containing one time column followed by one or more experimental traces.

Example:

```text
Time,Run_1,Run_2,Run_3
0.000,1.02,1.01,1.03
0.010,0.91,0.90,0.92
0.020,0.84,0.83,0.85
...
```

Requirements:

1. The **first column must contain time**.
2. All remaining columns are experimental traces.
3. Time is treated in **minutes**.
4. The number of trace columns must equal the total number of samples entered in the concentration-group setup.
5. Concentration groups are assigned to trace columns from left to right.

For example, entering:

- 50 µM, 9 samples
- 200 µM, 9 samples

means that the first nine trace columns are assigned to 50 µM and the next nine are assigned to 200 µM.

## Basic workflow

1. Launch `ANCHOR.py`.
2. Select the input CSV.
3. Choose an output parent directory and output folder name.
4. Enter the initial enzyme concentration, \(E_0\).
5. Define the substrate/reactant concentration groups and replicate counts.
6. Choose an analysis mode.
7. Review the fitting settings.
8. Select **Next: run**.
9. Run the fitting procedure.
10. Begin interpretation with:
   - `ANCHOR_key_results.csv`
   - `best_fit_summary.csv`
   - the fit-overlay figures

ANCHOR warns before writing into an output folder that already exists.

## Analysis modes

### General automatic model selection

Recommended for routine time-resolved kinetic datasets when the mechanism is not known in advance.

The default candidate library includes direct and multistep kinetic models together with linear-only and no-binding controls. Reversible Michaelis-Menten turnover is not automatically included in this mode.

### Auto-detect enzyme turnover

Compares:

\[
E + S \rightarrow P
\]

with:

\[
E + S \rightleftharpoons ES \rightarrow P
\]

while retaining linear-only and no-binding controls.

Use this mode for enzyme-turnover or progress-curve data when the appropriate turnover mechanism is not known beforehand.

### Known direct turnover only

Fits only:

\[
E + S \rightarrow P
\]

Use this mode when experimental evidence supports direct product formation without a resolvable bound intermediate.

### Known reversible MM only

Fits only:

\[
E + S \rightleftharpoons ES \rightarrow P
\]

and reports parameters associated with the reversible enzyme-substrate intermediate, including \(k_{\mathrm{on}}\), \(k_{\mathrm{off}}\), \(k_{\mathrm{cat}}\), and \(K_m\).

Use this mode when formation of an \(ES\) intermediate is supported independently by experiment.

### Custom advanced settings

Allows manual selection of the candidate models and detailed fitting parameters.

This is useful when independent experimental information justifies constraining the candidate library. For example, if binding is known from another method but the kinetic trace is approximately linear, the no-binding candidate can be disabled while retaining the linear-only model.

## Model selection

For each trace, ANCHOR screens the enabled candidate models and ranks them using information criteria. The best candidate is then refined using the full fitting window.

The candidate library can include:

- Direct 0-intermediate kinetic model
- Sequential models with 1–5 intermediates
- Slow-engagement models
- Reversible Michaelis-Menten model
- Linear-only model
- No-binding model

The no-binding model includes a noise-aware guardrail designed to reduce overinterpretation of small signal changes as mechanistic kinetic events.

Model selection should be interpreted together with fit quality, replicate consistency, and independent biochemical evidence. Selection of a more complex model does not by itself establish the physical identity of an intermediate.

## Michaelis-Menten analysis

When enabled, ANCHOR can collect selected turnover velocities across substrate concentrations and fit:

\[
v = \frac{V_{\max}[S]}{K_m + [S]}
\]

The Michaelis-Menten post-fit output may include:

- \(V_{\max}\)
- \(K_m\)
- apparent \(k_{\mathrm{cat}}\)
- \(k_{\mathrm{cat}}/K_m\)
- fit statistics
- individual velocity points
- concentration-group means
- a publication-ready Michaelis-Menten plot

The post-fit analysis requires sufficient positive finite turnover velocities at multiple concentrations.

## Main outputs

After fitting, ANCHOR creates an organized output directory. Key files include:

### `ANCHOR_key_results.csv`

A compact table containing the most important result for each trace, including the selected model and major fitted parameters.

### `best_fit_summary.csv`

A more detailed trace-level table containing fitted parameters and model statistics.

### `ANCHOR_summary.xlsx`

An Excel workbook collecting the major ANCHOR result tables.

### `run_configuration.json`

A record of the settings used for the analysis. This file is useful for reproducibility and reporting.

### `ANCHOR_output_guide.txt`

A short description of the generated files.

### Figures

Output figures are organized into folders that may include:

```text
figures/
├── per_trace_fits/
├── species/
├── overlays/
└── michaelis_menten/
```

### Michaelis-Menten outputs

When the MM post-fit option is enabled and appropriate turnover velocities are available, ANCHOR produces files such as:

```text
michaelis_menten_fit.png
michaelis_menten_fit_summary.csv
michaelis_menten_velocity_points.csv
michaelis_menten_grouped_velocity_summary.csv
```

## Included example datasets

Four examples are provided to demonstrate the major ANCHOR workflows.

### 1. Default model-selection example

File:

```text
examples/Default Model/ANCHOR_Default_Model_Test.csv
```

Setup:

- \(E_0 = 5.0\) µM
- 9 traces at 50 µM
- 9 traces at 200 µM
- Analysis mode: **General automatic model selection**

This example is intended to run successfully using the validated default settings.

### 2. Known-binding linear example

File:

```text
examples/Linear Binding/ANCHOR_Linear_Test.csv
```

Setup:

- \(E_0 = 5.0\) µM
- 9 traces at 50 µM
- 9 traces at 200 µM
- Analysis mode: **Custom advanced settings**
- Linear-only model enabled
- No-binding model disabled

Independent experiments support binding in this example, so the no-binding candidate is deliberately excluded.

### 3. No-binding example

File:

```text
examples/No Binding/ANCHOR_No_Binding_Test.csv
```

Setup:

- \(E_0 = 5.0\) µM
- 3 traces at 200 µM
- Analysis mode: **General automatic model selection**
- No-binding model enabled

### 4. Reversible Michaelis-Menten example

File:

```text
examples/MM Fitting/ANCHOR_MM_SPpNA_Fitting_Test.csv
```

Setup:

- \(E_0 = 40.0\) µM
- one trace at each substrate concentration:
  - 4500 µM
  - 3150 µM
  - 2250 µM
  - 1350 µM
  - 675 µM
  - 360 µM
- Analysis mode: **Known reversible MM only**
- Michaelis-Menten post-fit enabled

Formation of the \(ES\) intermediate is experimentally supported for this example, so the reversible mechanism is intentionally forced.

Detailed instructions are available in:

- `ANCHOR_Example_Testing_Guide.pdf`
- `ANCHOR_Example_Testing_Quick_Checklist.txt`

Preconfigured versions of these setups are available in `setup_configs/`.

## Reproducibility

For reproducible analyses:

- retain `run_configuration.json` with each analysis;
- report the ANCHOR version or Git commit used;
- retain the original input CSV;
- document any changes from the validated defaults;
- report which candidate models were enabled;
- inspect individual traces rather than relying only on aggregate model counts.

Because fitted kinetic parameters depend on the kinetic mechanism, observation model, fitting window, and candidate library, rate constants obtained from different fitting frameworks should only be treated as directly comparable when the underlying models are mathematically equivalent.

## Documentation

For additional information, see:

- **Quick start:** `ANCHOR_Quick_Start.txt`
- **Full user guide:** `ANCHOR_User_Guide.pdf`
- **Example testing guide:** `ANCHOR_Example_Testing_Guide.pdf`
- **Example checklist:** `ANCHOR_Example_Testing_Quick_Checklist.txt`

## Citation

A manuscript describing ANCHOR is currently being prepared for publication.

Until a formal citation is available, please cite the GitHub repository and include the version or commit hash used for the analysis. This section can be updated with the published ANCHOR citation once available.

## Reporting issues

Bug reports, feature requests, and reproducibility questions can be submitted through the GitHub repository's **Issues** page.

When reporting a fitting problem, please include:

- the ANCHOR version or commit hash;
- operating system and Python version;
- the relevant `run_configuration.json`;
- the error message or traceback;
- a minimal example dataset when it can be shared.

## Scientific use

ANCHOR is intended as a quantitative analysis tool. Automated model selection should complement, rather than replace, biochemical interpretation and independent experimental evidence. Candidate models describe the level of kinetic complexity supported by the observed trace; an ANCHOR-selected intermediate should not be assigned a specific molecular identity without supporting experimental evidence.

---

**ANCHOR — Automated Non-linear Curve fitting for High-speed Observed Reactions**
