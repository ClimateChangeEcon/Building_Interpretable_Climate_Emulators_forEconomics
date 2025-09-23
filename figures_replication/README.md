# Replication of the figures of the manuscript "Building Interpretable Climate Emulators for Economics"

* Each folder in this directory contains necessary scripts and data to reproduce respective figures presented in the paper.

* Upon running scripts the reproduced figures either stored in the same folder as *.png file or in the respective subfolders `figs_replication`.

* The data for the figures 3-4, 14-25 is located and explained in the folder
[data](../data). Calibration results that are used for plotting are presented in the folder [result](figures_3-4_14-25/result). The instructions how to reproduce the calibration results are in [README](figures_3-4_14-25/README.md) .

* The data for the figures 5-8, 26-31 is being produced from the solutions of
the models and can be found here [runs](../DEQN/runs). Replication code for the solutions can be found in the folder [DEQN](../DEQN).

* To replicate the figures please make sure you are in the folder `/figures_replication` and do following:

---

**Fig.3, Fig. 23**

Go to the folder `figures_3-4_14-25`. Run the script `fig_benchmark_pulse.py`. This file requires the additional files ``lib_data_load.py``, ``lib_models.py``, and ``lib_operations.py`` which are also located in the same directory. 

```
python fig_benchmark_pulse.py
```

The resulting figures 3, 23 will be sotred in the folder [figs_replication](figures_3-4_14-25/figs_replication).

---

**Fig.4, Fig.16, Fig. 17**

Go to the folder `figures_3-4_14-25`. Run the script `fig_analysis.py`. This file requires the additional files ``lib_data_load.py``, ``lib_models.py``, and ``lib_operations.py`` which are also located in the same directory. 

```
python fig_analysis.py
```

The resulting figures 4, 16, 17 will be sotred in the folder [figs_replication](figures_3-4_14-25/figs_replication).

---

**Fig.5-8**

Go to the folder `figures_5-8_26-31`. Run the jupyter-notebook `figs_5-8.ipynb`. 

The resulting figures 5-8 will be shown in the notebook as well as stored in the folder [figs_replication](figures_5-8_26-31/figs_replication).

Note, that the code rpelicatiuon the figures 5-8 also contains an additional code to replicate tables 3-6. The resulting tables in *.tex format are also stored to the same folder as figures.

---

**Fig.9**

Go to the folder `figure_9`. Run the script `figure_9.py`. This file requires the additional data which are also located in the same directory. 

```
python figure_9.py
```
The components of the figure 9 will be stored in the same folder as the code.

---

**Fig.10**

Go to the folder `figure_10`. Run the jupyter notebook `Figure_10_WGI_v4_regions_from_CSV_land_and_landocean.ipynb`. This file requires the additional data which are also located in the same directory. 

All the figures produced by the script are stored in the folder `fig`. The reproduced `figure_10.png` will be sotred in the folder [figs_replication](figure_10/figs_replication). 

---

**Fig.11**

Go to the folder `figure_11`. Run the script `figure_11.py`. This file requires the additional data which are also located in the same directory. 

```
python figure_11.py
```

All the figures produced by the script are stored in the folder `results_pattern_2p5_multi`. The reproduced Figure 11 will be sotred in the folder [figs_replication](figure_11/figs_replication). 

---

**Fig.12**

Go to the folder `figure_12`. Run the script `figure_12.py`. This file requires the additional data which are also located in the same directory. 

```
python figure_12.py
```
The figure 12 will be stored in the same folder as the code.

---

**Fig.13**

Go to the folder `figure_13`. Run the script `figure_13.py`.

```
python figure_13.py
```
The figure 13 will be stored in the same folder as the code.

---

**Fig.14**

Go to the folder `figures_3-4_14-25`. Run the script `fig_RCP_RF.py`. This file requires the additional files ``lib_data_load.py``, ``lib_models.py``, and ``lib_operations.py`` which are also located in the same directory. 

```
python fig_RCP_RF.py
```

The resulting `figure_14.png` will be sotred in the folder [figs_replication](figures_3-4_14-25/figs_replication).

---

**Fig.15, Fig.18-Fig.22, Fig.24, Fig.25**

Go to the folder `figures_3-4_14-25`. Run the script `fig_sim.py`. This file requires the additional files ``lib_data_load.py``, ``lib_models.py``, and ``lib_operations.py`` which are also located in the same directory. 

```
python fig_sim.py
```

The resulting figures 15, 18-22, 24, 25 will be sotred in the folder [figs_replication](figures_3-4_14-25/figs_replication).

---

**Fig.26-27**

Go to the folder `figures_5-8_26-31`. Run the jupyter-notebook `figs_26-27.ipynb`. 

The resulting figures 26-27 will be shown in the notebook as well as stored in the folder [figs_replication](figures_5-8_26-31/figs_replication).

---

**Fig.28-29**

Go to the folder `figures_5-8_26-31`. Run the jupyter-notebook `figs_28-29.ipynb`. 

The resulting figures 28-29 will be shown in the notebook as well as stored in the folder [figs_replication](figures_5-8_26-31/figs_replication).

---

**Fig.30-31**

Go to the folder `figures_5-8_26-31`. Run the jupyter-notebook `figs_30-31.ipynb`. 

The resulting figures 30-31 will be shown in the notebook as well as stored in the folder [figs_replication](figures_5-8_26-31/figs_replication).

---

**Fig.32**

Go to the folder `figure_32`. Run the script `figure_32.py`. This file requires the additional data which are also located in the same directory. 

```
python figure_32.py
```
The figure 32 will be stored in the same folder as the code.



