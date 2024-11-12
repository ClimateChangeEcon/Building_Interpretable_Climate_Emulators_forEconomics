# Building Interpretable Climate Emulators for Economics

This Python-based code repository supplements the work of [Aryan Eftekhari](https://scholar.google.com/citations?user=GiugKBsAAAAJ&hl=en), [Doris Folini](https://iac.ethz.ch/people-iac/person-detail.NDY3MDg=.TGlzdC82MzcsLTE5NDE2NTk2NTg=.html), [Aleksandra Friedl](https://sites.google.com/view/aleksandrafriedl?pli=1), [Felix Kübler](https://sites.google.com/site/fkubler/), [Simon Scheidegger](https://sites.google.com/site/simonscheidegger/), and [Olaf Schenk](https://search.usi.ch/en/people/9a52a2fdb8d3d26ec16fb1569b590909/schenk-olaf), titled _[Building Interpretable Climate Emulators for Economics](#citation)_ (Eftekhari et al.; 2024).
It presents a flexible computational framework that enables IAM modelers to create interpretable, efficient, and custom-tailored climate emulators to address specific research questions, even without advanced climate science expertise.

<Write breif overview of project>


## Carbon Cycle Models
This section provides a formal description of a linear multi-reservoir carbon cycle model. It discusses the limitations and challenges of using a linear model to emulate the complexities of physical non-linear dynamics in more sophisticated models, such as Earth System Models (ESMs) or Earth System Models of Intermediate Complexity (ESMIC), and to capture the variability of carbon cycle responses across these models. The focus is on two carbon cycle model configurations: serial and parallel reservoir configurations. The models include three reservoir classes: atmosphere (A), ocean (O), and land biosphere (L). The serial model (3SR) consists of three sequentially connected carbon reservoirs, with the atmosphere connected to the upper ocean (O<sub>1</sub>), and the ocean connected to the deep ocean (O<sub>2</sub>). The parallel model (4PR) introduces the land biosphere, where carbon from the atmosphere is divided into two parallel streams: land biosphere and ocean.

![image](https://drive.google.com/uc?id=1HPtr5Wff0OOALSiU70ZoafYnTt_qG5xk)

The operator $\mathbf{A}$ is visualized for the 4PR model (left), which includes atmosphere (A), two ocean reservoirs (O<sub>1</sub> and O<sub>2</sub>), and a land reservoir (L). A graphical representation of the connectivity of the reservoirs (right) is shown with the unknown carbon mass transfer rates identified, for example, O<sub>1</sub>  $\to$ O<sub>2</sub> corresponds to the entry $\mathbf{A}_{3,2}$. $\mathbf{A}$ is symmetric in its nonzero pattern, but not in its values.


## Workflow

### 1. Model Fitting (x_solver.ipynb)
- Steps to load data, fit models, and optimize parameters.

### 2. Fitted Model Analysis & Calibration (x_analyse.ipynb):
- Tools for visualizing and calibrating model performance.

### 3. Model Simulation and Comparisons (x_simulate.ipynb):
- Guidelines for setting up simulations, comparing outputs, and analyzing results.

## Support Libraries (lib_*)
This repository contains tools and scripts for climate modeling, simulation, and analysis. It includes data handling, model definitions, operational functions, and interactive notebooks for detailed climate studies.

### lib_data_load.py: Data and Benchmark Datasets
- **load_data**: Load climate data from various sources.
- **preprocess_data**: Clean and prepare data for model consumption.
- **benchmark_datasets**: Manage benchmark datasets for validation.

### lib_models.py: Climate Models
- **initialize_model**: Set up initial conditions for models.
- **run_model**: Execute climate model simulations.
- **evaluate_model**: Assess model performance with various metrics.

### lib_operations.py: Operations Models
- **apply_operations**: Perform a series of operations on model data.
- **optimize_parameters**: Refine model parameters for better accuracy.
- **validate_results**: Compare outputs with benchmarks for validation.


## Getting Started

1. **Clone the repository:**
   ```sh
   git clone git@github.com:ClimateChangeEcon/Building_Interpretable_Climate_Emulators_forEconomics.git
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Explore the notebooks:**
   - Use `x_solver.ipynb` to fit models.
   - Use `x_analyse.ipynb` to analyze fitted models.
   - Use `x_simulate.ipynb` to run simulations and compare results.


## Citation

Please cite [Building Interpretable Climate Emulators for Economics, A. Eftekhari, D. Folini, A. Friedl, F. Kübler, S. Scheidegger, O. Schenk](https://epubs.siam.org/doi/10.1137/21M1392231) in your publications if it helps your research:
```
@article{XXXX,
    title={Building Interpretable Climate Emulators for Economics},
    author={Aryan Eftekhari and Doris Folini and Aleksandra Friedl and Felix Kübler and Simon Scheidegger and Olaf Schenk},
    year={2024}
}
```
See [here](https://arxiv.org/pdf/XXX.pdf) for an archived version of the article. 


### Authors
* [Aryan Eftekhari](https://scholar.google.com/citations?user=GiugKBsAAAAJ&hl=en) (Department of Informatics, Institute of Computing, Università della Svizzera italiana)
* [Doris Folini](https://iac.ethz.ch/people-iac/person-detail.NDY3MDg=.TGlzdC82MzcsLTE5NDE2NTk2NTg=.html) (Department of Environmental Systems Science, ETH Zurich)
* [Aleksandra Friedl](https://sites.google.com/view/aleksandrafriedl?pli=1) (ifo Center for Energy, Climate and Resources)
* [Felix Kübler](https://sites.google.com/site/fkubler/) (Department of Finance, University of Zurich)
* [Simon Scheidegger](https://sites.google.com/site/simonscheidegger/) (Department of Economics, University of Lausanne)
* [Olaf Schenk](https://search.usi.ch/en/people/9a52a2fdb8d3d26ec16fb1569b590909/schenk-olaf) (Department of Informatics, Institute of Computing, Università della Svizzera italiana)

### Other Relate Research
* [The Climate in Climate Economics; Folini, D; Friedl, A.; Kubler, F; Scheidegger, S (2024)](https://academic.oup.com/restud/advance-article-abstract/doi/10.1093/restud/rdae011/7593489?redirectedFrom=fulltext&login=false).


## Contributing

We welcome contributions! Please fork this repository, make your changes, and submit a pull request.


## License

This project is licensed under the MIT License. See the LICENSE file for details.


## Support
This work is generously supported by grants from the [Swiss National Science Foundation](https://www.snf.ch) under project IDs “New methods for asset pricing with frictions”, "Can economic policy mitigate climate change", the [Enterprise for Society (E4S)](https://e4s.center), and Emmanuel Jeanvoine from UNIL's [DCSR](https://www.unil.ch/ci/fr/home/menuinst/calcul--soutien-recherche.html).
