# Climate Emulator Repository

This folder contains Python scripts for building, fitting, and visualizing climate emulators based on the framework described in the paper. The workflow is designed for Integrated Assessment Model (IAM) practitioners to create interpretable and efficient carbon cycle emulators.

For visualizations:
 - ```python3 fig_benchmark_pulse.py``` Plot benchmark pulse decays for different models
 - ```python3 fig_RCP_RF.py```  Plots related to the RCP scenerios
 - ```python3 fig_analysis.py``` Analyse emulator hyperparameters and solve extrema paramters for the MMM fitted emulator.
 - ```python3 fig_sim.py``` Different simulations using the extrama paramater of MMM fitted emulator.
---

## Requirements

* Python 3.8+, numpy, scipy, matplotlib, prettytable, pickle, logging, multiprocessing 

---

## Scripts Overview

### 1. `solver.py`

**Purpose:** Fits the 3SR and 4PR carbon cycle models to benchmark datasets, performing a grid search over penalty parameters.

**Usage:**

```bash
python3 solver.py <conditions: PI|PD> <T:int> <benchmark_sel>
```

* **PI**: Preindustrial benchmark set
* **PD**: Present-day benchmark set
* `<T>`: Time horizon in years
* `<benchmark_sel>`: One benchmark model name from the allowed set for the selected condition

**Example:**

```bash
python3 solver.py PD 250 CLIMBER2
```

Each folder contains:

* `opt.log`: Optimization log
* `result.pkl`: Fitted model parameters and metadata
* `opt_data.pkl`: Objective and penalty term histories

---

### 2. `fig_benchmark_pulse.py`

**Purpose:** Plots benchmark impulse response (pulse) data for selected models.

**Usage:**

```bash
python3 fig_benchmark_pulse.py
```

This script will generate figures and save them in the `fig/` directory.

---

### 3. `fig_RCP_RF.py`

**Purpose:** Plots Representative Concentration Pathway (RCP) scenarios and radiative forcing data.

**Usage:**

```bash
python3 fig_RCP_RF.py
```

This script produces RCP emissions, cumulative emissions, CO₂ forcing, and forcing ratio plots.

---

### 4. `fig_analysis.py`

**Purpose:** Performs post-fit analysis of emulator results.

**Usage:**

```bash
python3 fig_analysis.py
```

Reads fit results from the output directories and generates performance/diagnostic plots.

---

### 5. `fig_sim.py`

**Purpose:** Runs and visualizes emulator simulations for different scenarios.

**Usage:**

```bash
python3 fig_sim.py
```

Generates simulation figures for 3SR and 4PR configurations.

