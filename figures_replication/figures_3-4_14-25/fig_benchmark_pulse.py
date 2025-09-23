import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import logging
import os
import sys
import copy
import pickle

import lib_operations as operations
import lib_data_load as data_load
# from lib_models_fixed import model_3sr,model_4pr
from lib_models import model_3sr, model_4pr

_plot_vis = {}
_plot_vis['color'] = ['black', 'red', 'blue',
                      'green', 'orange', 'purple', 'brown', 'pink']
_plot_vis['style'] = ['solid', 'dashed', 'dotted', 'dashdot']
_plot_vis['marker'] = ['o', 'v', 's', 'D', 'X']


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

_conditions = 'PD'
_T = 250


def plot_pulse(conditions, T_set=[500, 100]):
    fig, axs = make_plot(1, 2)

    for inx, T in enumerate(T_set):

        if conditions == 'PI':
            benchmark_list = ['NCAR', 'BERN3D', 'BERN25D', 'CLIMBER2', 'DCESS','GENIE', 'LOVECLIM', 'MESMO', 'UVIC29', 'BERNSAR', 'MMM', 'MMMU', 'MMMD']

        elif conditions == 'PD':
            benchmark_list = ['NCAR', 'HADGEM2', 'MPIESM', 'BERN3DR', 'BERN3DE', 'BERN25D', 'CLIMBER2', 'DCESS','GENIE', 'LOVECLIM', 'MESMO', 'UVIC29', 'ACC2', 'BERNSAR', 'MAGICC6', 'TOTEM2', 'MMM', 'MMMU', 'MMMD']
        else:
            raise NameError('Invalid _conditions')

        for i, benchmark_i in enumerate(benchmark_list[0:-1]):
            [d, name] = data_load.pulse_fraction(test_type=benchmark_i, conditions=conditions, T=T)

            print(name)
            d = d*1
            t = min(1000, len(d))
            axs[0, inx].plot(range(0, len(d)), d, linewidth=1,alpha=.2, color='black', linestyle='-')

        benchmark_list = ['MMM', 'MMMD', 'MMMU', 'CLIMBER2', 'MESMO']
        color_set = ['black', 'blue', 'red', 'magenta', 'orange']
        line_style_set = ['-', '--', ':', '-.', '--']
        for i, benchmark_i in enumerate(benchmark_list):
            [d, name] = data_load.pulse_fraction(test_type=benchmark_i, conditions=conditions, T=T)
            d = d*1
            t = min(1000, len(d))
            axs[0, inx].plot(range(0, len(d)), d,   color=color_set[i],label=name, linewidth=2, alpha=1, linestyle=line_style_set[i])
            # axs[0,inx].plot(range(0,len(d)),d,   color=color_set[i],label=benchmark_i,linewidth=2,alpha=1,linestyle=line_style_set[i])

        [d, name] = data_load.pulse_fraction(test_type=benchmark_list[-1], conditions=conditions, T=T)
        axs[0, inx].plot(range(0, len(d)), d, linewidth=1, alpha=.2,color='black', linestyle='-', label='Test Models')

        axs[0, inx].minorticks_on()
        axs[0, inx].set_xlim(0, T)
        axs[0, inx].set_ylim(.0, 1)

    axs[0, 0].set_ylabel(r'Pulse Fraction')
    axs[0, 0].set_xlabel(r'$t$ (Years)')
    axs[0, 1].set_xlabel(r'$t$ (Years)')
    axs[0, 0].legend(frameon=False, ncol=2)

    fig_path = f'fig/{conditions}/solver_pulse.png'
    fig.savefig(fig_path, dpi=300)
    print(f"[SAVE] Plot saved to: {fig_path}")
    if conditions=='PD':
        fig.savefig('figs_replication/figure_23.png', dpi=300)
    if conditions=='PI':
        fig.savefig('figs_replication/figure_3.png', dpi=300)


plot_pulse(conditions='PI')
plot_pulse(conditions='PD')
