import os
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import copy
import pickle

import lib_operations as operations
import lib_data_load as data_load
from lib_models import model_3sr, model_4pr


def make_plot(n, m, s=[1, 1]):
    if n == 1:
        return plt.subplots(n, m, figsize=(m*4*s[0], n*2.8*s[1]), squeeze=False, layout='tight')
    else:
        return plt.subplots(n, m, figsize=(m*4*s[0], n*2.6*s[1]), squeeze=False, layout='tight')

np.set_printoptions(suppress=True)
plt.rcParams['text.usetex'] = True
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 11})
plt.rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{amsmath}"

fig_1, axs_1 = make_plot(2, 2)
fig_2, axs_2 = make_plot(1, 2)
T_start = 1765
T_end = 2100

scenerio_set_name = ['RCP 2.6', 'RCP 4.5', 'RCP 6.0', 'RCP 8.5']
[reactive_forcing_co2, reactive_forcing_ratio, reactive_forcing_year] = data_load.reactive_forcing(T_start=T_start, T_end=T_end)

color_set = ['black', 'blue', 'red', 'green', 'orange']
linestyle_set = ['-', '--', ':', '-.', '-']

for i, scenerio in enumerate(['RCP2.6', 'RCP4.5', 'RCP6.0', 'RCP8.5']):
    [data_val, data_year] = data_load.cmip_emission(scenerio_name=scenerio, T_start=T_start, T_end=T_end, emission_type='fossil')
    axs_1[0, 0].plot(data_year, data_val, label=scenerio_set_name[i], linewidth=2, color=color_set[i], linestyle=linestyle_set[i])
    axs_1[1, 0].plot(data_year, np.cumsum(data_val), label=scenerio_set_name[i], linewidth=2, color=color_set[i], linestyle=linestyle_set[i])

    [data_val, data_year] = data_load.cmip_emission(scenerio_name=scenerio, T_start=T_start, T_end=T_end, emission_type='land')
    axs_1[0, 1].plot(data_year, data_val, label=scenerio_set_name[i], linewidth=2, color=color_set[i], linestyle=linestyle_set[i])
    axs_1[1, 1].plot(data_year, np.cumsum(data_val), label=scenerio_set_name[i], linewidth=2, color=color_set[i], linestyle=linestyle_set[i])

    axs_2[0, 0].plot(data_year, reactive_forcing_co2[scenerio], label=scenerio_set_name[i], linewidth=2, color=color_set[i], linestyle=linestyle_set[i])
    axs_2[0, 1].plot(data_year, reactive_forcing_ratio[scenerio], label=scenerio_set_name[i], linewidth=2, color=color_set[i], linestyle=linestyle_set[i])

axs_2[0, 0].plot(data_year, reactive_forcing_co2['avg'], label='Mean', linewidth=1, color='magenta', linestyle='-')
axs_2[0, 1].plot(data_year, reactive_forcing_ratio['avg'], label='Mean', linewidth=1, color='magenta', linestyle='-')

# Formatting and labels
axs_2[0, 0].minorticks_on()
axs_2[0, 1].minorticks_on()
for ax_row in axs_1:
    for ax in ax_row:
        ax.minorticks_on()

axs_1[0, 0].legend(frameon=False, loc='upper left', ncol=1)
axs_2[0, 1].legend(frameon=False, loc='lower right', ncol=1)

axs_1[0, 0].set_ylabel(r'Emissions (GtC/yr)')
axs_1[1, 0].set_ylabel(r'Cumulative Emissions (GtC)')
axs_2[0, 0].set_ylabel(r'$\mathcal{F}_{2\times\text{CO}_2}$ ($\textrm{W} \textrm{m}^{-2}$)')
axs_2[0, 1].set_ylabel(r'$\mathcal{F}_{2\times \text{Total}} / \mathcal{F}_{2\times\text{CO}_2}$')

axs_1[1, 0].set_xlabel(r'Year')
axs_1[1, 1].set_xlabel(r'Year')
axs_2[0, 0].set_xlabel(r'Year')
axs_2[0, 1].set_xlabel(r'Year')

# Save figures with directory creation and logging
fig1_path = 'fig/simulations_RCP.png'
os.makedirs(os.path.dirname(fig1_path), exist_ok=True)
fig_1.savefig(fig1_path, dpi=300)
fig_1.savefig('figs_replication/figure_14.png', dpi=300)
print(f"[SAVE] RCP plot saved to: {fig1_path}")

fig2_path = 'fig/simulations_RF.png'
os.makedirs(os.path.dirname(fig2_path), exist_ok=True)
fig_2.savefig(fig2_path, dpi=300)
print(f"[SAVE] RF plot saved to: {fig2_path}")