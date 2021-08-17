import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # avoids jax gpu warning
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision

import re
import copy
import argparse
import ast
import numpy as np
import pandas as pd
import seaborn as sns
import jax.numpy as jnp
from jax import random
from collections import defaultdict

import matplotlib
import matplotlib.transforms
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf
from matplotlib.colors import ListedColormap

from dibs.eval.result import fields_distributionEvaluation_metrics

PLOT_STORE_ROOT = ['plots']

PLOT_ID_LIMS = {
    'bge20' : {
        'top':    {0: (None, 60), 1: (0.60, 1.00), 2: (None, None), 3: (None, None),},
        'bottom': {0: (None, 52), 1: (0.60, 1.00), 2: (None, 980), 3: (None, None),},
        'single': {0: (None, None), 1: (None, None), 2: (None, None), 3: (None, None),},
    },
    'bge50' : {
        'top':    {0: (None, None), 1: (None, None), 2: (None, None), 3: (None, None),},
        'bottom': {0: (None, None), 1: (0.6, None), 2: (None, None), 3: (None, None),},
        'single': {0: (None, None), 1: (None, None), 2: (None, None), 3: (None, None), },
    },
    'lingauss20' : {
        'top':    {0: (8, 80), 1: (0.53, 1.0), 2: (400, 2800), 3: (300, 2900),},
        'bottom': {0: (8, 60),  1: (0.65, 1.0), 2: (400, 2000), 3: (300, 1800),},
        'single': {0: (None, None), 1: (None, None), 2: (None, None), 3: (None, None),},
    },
    'lingauss50' : {
        'top':    {0: (None, None), 1: (None, None), 2: (None, 8500), 3: (None, None),},
        'bottom': {0: (None, None), 1: (None, None), 2: (None, 8500), 3: (None, 6500),},
        'single': {0: (None, None), 1: (None, None), 2: (None, None), 3: (None, None),},
    },
    'fcgauss20' : {
        'top':    {0: (None, None), 1: (None, None), 2: (-10000, 150000), 3: (-10000, 150000), },
        'bottom': {0: (12, 57),     1: (None, None), 2: (-10000, 120000), 3: (-10000, 120000), },
        'single': {0: (None, None), 1: (None, None), 2: (None, None), 3: (None, None),},
    },
    'fcgauss50' : {
        'top':    {0: (None, 550), 1: (None, None), 2: (-50000, 500000), 3: (-50000, 950000),},
        'bottom': {0: (None, 350), 1: (None, None), 2: (-50000, 300000), 3: (-50000,  600000), },
        'single': {0: (None, None), 1: (None, None), 2: (None, None), 3: (None, None),},
    },
# ablation 
    'lingauss20-ablation-embed' : {
        'single':    {0: (10, 70), 1: (0.65, 1.00), 2: (200, 3800), 3: (200, 3000),},
    },
    'lingauss20-ablation-dim' : {
        'single':    {0: (None, None), 1: (0.73, 1.00), 2: (200, 3500), 3: (200, 2900), },
    },
    'lingauss20-ablation-steps': {
        'single':    {0: (None, None), 1: (0.70, 1.00), 2: (200, 4500), 3: (200, 3200), },
    },
}

COLOR_SATURATION = 0.8

DPI = 300

LINE_WIDTH = 7.0
COL_WIDTH = 3.333

FIG_SIZE_TRIPLE = (COL_WIDTH / 3, COL_WIDTH / 3 * 4/6)
FIG_SIZE_TRIPLE_TALL = (COL_WIDTH / 3, COL_WIDTH / 3 * 5/6)

FIG_SIZE_DOUBLE = (COL_WIDTH / 2, COL_WIDTH / 2 * 4/6)
FIG_SIZE_DOUBLE_TALL = (COL_WIDTH / 2, COL_WIDTH / 2 * 5/6)

CUSTOM_FIG_SIZE_FULL_PAGE_TRIPLE = (LINE_WIDTH / 3, COL_WIDTH / 2 * 5/6)

FIG_SIZE_FULL_PAGE_TRIPLE = (LINE_WIDTH / 3, LINE_WIDTH / 3 * 4/6)
FIG_SIZE_FULL_PAGE_TRIPLE_TALL = (LINE_WIDTH / 3, LINE_WIDTH / 3 * 5/6)

CUSTOM_FIG_SIZE_FULL_PAGE_QUAD = (LINE_WIDTH / 4, COL_WIDTH / 2 * 5/6)


NEURIPS_LINE_WIDTH = 5.5  # Text width: 5.5in (double figure minus spacing 0.2in).
FIG_SIZE_NEURIPS_DOUBLE = (NEURIPS_LINE_WIDTH / 2, NEURIPS_LINE_WIDTH / 2 * 4/6)
FIG_SIZE_NEURIPS_TRIPLE = (NEURIPS_LINE_WIDTH / 3, NEURIPS_LINE_WIDTH / 3 * 4/6)
FIG_SIZE_NEURIPS_DOUBLE_TALL = (NEURIPS_LINE_WIDTH / 2, NEURIPS_LINE_WIDTH / 2 * 5/6)
FIG_SIZE_NEURIPS_TRIPLE_TALL = (NEURIPS_LINE_WIDTH / 3, NEURIPS_LINE_WIDTH / 3 * 5/6)

NEURIPS_RCPARAMS = {
    "figure.autolayout": True,       # `False` makes `fig.tight_layout()` not work
    "figure.figsize": FIG_SIZE_NEURIPS_DOUBLE,
    # "figure.dpi": DPI,             # messes up figisize
    # Axes params
    "axes.linewidth": 0.5,           # Matplotlib's current default is 0.8.
    "xtick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    "ytick.major.width": 0.5,
    "ytick.minor.width": 0.5,

    "hatch.linewidth": 0.3,
    "xtick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    'xtick.major.pad': 3.0, 
    'xtick.major.size': 1.75,
    'xtick.minor.pad': 1.0,
    'xtick.minor.size': 1.0,

    'ytick.major.pad': 1.0,
    'ytick.major.size': 1.75,
    'ytick.minor.pad': 1.0,
    'ytick.minor.size': 1.0,

    "axes.labelpad": 0.5,
    # Grid
    "grid.linewidth": 0.3,
    # Plot params
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    'errorbar.capsize': 3.0,
    # Font
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",             # use serif rather than sans-serif
    "font.serif": "Times New Roman",    # use "Times New Roman" as the standard font
    "font.size": 8.5,
    "axes.titlesize": 8.5,                # LaTeX default is 10pt font.
    "axes.labelsize": 8.5,                # LaTeX default is 10pt font.
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    # Legend
    "legend.fontsize": 7,        # Make the legend/label fonts a little smaller
    "legend.frameon": True,              # Remove the black frame around the legend
    "legend.handletextpad": 0.3,
    "legend.borderaxespad": 0.2,
    "legend.labelspacing": 0.1,
    "patch.linewidth": 0.5,
    # PDF
    "pgf.texsystem": "xelatex",         # use Xelatex which is TTF font aware
    "pgf.rcfonts": False,               # Use pgf.preamble, ignore standard Matplotlib RC
    "pgf.preamble": (
        r'\usepackage{fontspec}'
        r'\usepackage{unicode-math}'
        r'\setmainfont{Times New Roman}'
    ),
    "text.latex.preamble": (
        r'\usepackage{amsmath}'
        r'\usepackage{amssymb}'
        r'\usepackage{amsfonts}'
    ),
}


metric_format_dict = {
    'kl_q_p' : r'$KL(q||p)$',
    'kl_p_q': r'$KL(q||p)$',
    'mmd_shd_01_sample': r'$MMD(p, q)$',
    'mmd_shd_1_sample': r'$MMD(p, q)$',
    'mmd_shd_10_sample':  r'$MMD(p, q)$',
    'l1_edge_belief': r'$L_1$ edge belief',
    'expected_shd': r'$\mathbb{E}$-SHD',
    'expected_shd_cpdag': r'$\mathbb{E}$-SHD',
    'best_train_neg_log_marginal_likelihood':       'best train neg. MLL',
    'best_train_neg_log_likelihood' :               'best train neg. LL',
    'neg_ave_train_marginal_likelihood':            'neg. train MLL',
    'neg_ave_train_likelihood':                     'neg. train LL',
    'neg_train_log_posterior_predictive':           'neg. train log PP',
    'neg_ave_test_log_marginal_likelihood':         'neg. MLL',
    'neg_ave_test_log_likelihood':                  'neg. LL',
    'neg_test_log_posterior_predictive':            'neg. log PP',
    'neg_ave_interv_log_marginal_likelihood':       'neg. I-MLL',
    'neg_ave_interv_log_likelihood':                'neg. I-LL',
    'neg_interv_log_posterior_predictive':          'neg. log I-PP',
    # 'best_train_neg_log_marginal_likelihood':       'best train neg. MLL',
    # 'best_train_neg_log_likelihood' :               'best train neg. LL',
    # 'neg_ave_train_marginal_likelihood':            'neg. train MLL',
    # 'neg_ave_train_likelihood':                     'neg. train LL',
    # 'neg_train_log_posterior_predictive':           'neg. train log PP',
    # 'neg_ave_test_log_marginal_likelihood':         'neg. held-out MLL',
    # 'neg_ave_test_log_likelihood':                  'neg. held-out LL',
    # 'neg_test_log_posterior_predictive':            'neg. held-out log PP',
    # 'neg_ave_interv_log_marginal_likelihood':       'neg. interventional MLL',
    # 'neg_ave_interv_log_likelihood':                'neg. interventional LL',
    # 'neg_interv_log_posterior_predictive':          'neg. interventional log PP',
    'roc_auc': r'AUROC',
    'prc_auc': r'AUPRC',
    'ave_prec': r'AP',
    'n_unique_graphs': r'\# unique graphs',
    'n_cyclic_graphs': r'\# cyclic graphs',
    'walltime': 'execution time',
}


metric_scale_dict = {
    'kl_q_p': 'log',
    'kl_p_q': 'log',
    'mmd_shd_01_sample': 'log',
    'mmd_shd_1_sample': 'log',
    'mmd_shd_10_sample':  'log',
    'l1_edge_belief' : 'linear',
    'expected_shd' : 'linear',
    'expected_shd_cpdag' : 'linear',
    'best_train_neg_log_marginal_likelihood': 'linear',
    'best_train_neg_log_likelihood': 'linear',
    'neg_ave_train_marginal_likelihood': 'linear',
    'neg_ave_train_likelihood': 'linear',
    'neg_train_log_posterior_predictive': 'linear',
    'neg_ave_test_log_marginal_likelihood': 'linear',
    'neg_ave_test_log_likelihood': 'linear',
    'neg_test_log_posterior_predictive': 'linear',
    'neg_ave_interv_log_marginal_likelihood': 'linear',
    'neg_ave_interv_log_likelihood': 'linear',
    'neg_interv_log_posterior_predictive': 'linear',
    'roc_auc': 'linear',
    'prc_auc': 'linear',
    'ave_prec': 'linear',
    'n_unique_graphs' : 'linear',
    'n_cyclic_graphs' : 'linear',
    'walltime': 'log',
}

method_format_dict = {
    'mcmc_structure': r'MC${}^3$',
    'mh_joint_mcmc_structure': r'Metropolis-Hastings MC${}^3$',
    'gibbs_joint_mcmc_structure': r'MH-within-Gibbs MC${}^3$',
    'boot_ges' : 'GES Bootstrap',
    'boot_pc' : 'PC Bootstrap',
    'joint_boot_ges' : 'GES Bootstrap + MLE',
    'joint_boot_pc' : 'PC Bootstrap + MLE',

    'dibs_empirical' :          'DiBS  (ours)',
    'dibs_mixture':             'DiBS+ (ours)',
    'dibs_empirical_embed_off' :'Scalar',
    'dibs_mixture_embed_off':   'Scalar (mixture)',
    'dibs_empirical_embed_on':  'Inner Product',
    'dibs_mixture_embed_on':    'Inner Product (mixture)',

    "dibs_empirical_ndim=5":        r'$k=5$',
    "dibs_mixture_ndim=5":          r'$k=5$  (mixture)',
    "dibs_empirical_ndim=7":        r'$k=7$',
    "dibs_mixture_ndim=7":          r'$k=7$  (mixture)',
    "dibs_empirical_ndim=10":       r'$k=10$',
    "dibs_mixture_ndim=10":         r'$k=10$ (mixture)',
    "dibs_empirical_ndim=15":       r'$k=15$',
    "dibs_mixture_ndim=15":         r'$k=15$ (mixture)',
    "dibs_empirical_ndim=20":       r'$k=20$',
    "dibs_mixture_ndim=20":         r'$k=20$ (mixture)',
    "dibs_empirical_ndim=30":       r'$k=30$',
    "dibs_mixture_ndim=30":         r'$k=30$ (mixture)',
    "dibs_empirical_ndim=50":       r'$k=50$',
    "dibs_mixture_ndim=50":         r'$k=50$ (mixture)',

    "dibs_empirical_steps=100":      r'100 steps',
    "dibs_mixture_steps=100":        r'100 steps',
    "dibs_empirical_steps=200":      r'200 steps',
    "dibs_mixture_steps=200":        r'200 steps',
    "dibs_empirical_steps=400":      r'400 steps',
    "dibs_mixture_steps=400":        r'400 steps',
    "dibs_empirical_steps=800":      r'800 steps',
    "dibs_mixture_steps=800":        r'800 steps',
    "dibs_empirical_steps=1600":     r'1,600 steps',
    "dibs_mixture_steps=1600":       r'1,600 steps',
    "dibs_empirical_steps=3200":     r'3,200 steps',
    "dibs_mixture_steps=3200":       r'3,200 steps',
}


row_order_dict = {
    'graph_dist': ['er', 'sf'],
}

row_format_dict = {
    'er':  r'Erd{\H{o}}s-R{\'e}nyi',
    'sf' : r'Scale-free'
}


# https://colorbrewer2.org/#type=qualitative&scheme=Set1&n=9


sequential_green = [
    '#edf8e9',
    '#bae4b3',
    '#74c476',
    '#31a354',
    '#006d2c',
]

sequential_red = [
    '#fee5d9',
    '#fcae91',
    '#fb6a4a',
    '#de2d26',
    '#a50f15',
]

sequential_red7 = [
    '#fee5d9',
    '#fcbba1',
    '#fc9272',
    '#fb6a4a',
    '#ef3b2c',
    '#cb181d',
    '#99000d',
]


on_off_green = [
    sequential_green[1],
    sequential_green[4],
]

on_off_red = [
    sequential_red[1],
    sequential_red[4],
]

method_colors = {
    'boot_ges':                     '#377eb8',
    'boot_pc':                      '#4daf4a',
    'joint_boot_ges':               '#377eb8',  # same as marginal
    'joint_boot_pc':                '#4daf4a',  # same as marginal

    'mcmc_structure':               '#984ea3',  # lighter: #decbe4
    'mh_joint_mcmc_structure':      '#984ea3',  # prev: #f781bf
    'gibbs_joint_mcmc_structure':   '#999999',  # prev: #999999

    'dibs_empirical':               '#ff7f00',
    'dibs_mixture':                 '#e41a1c',

    'dibs_empirical_embed_off':     on_off_red[0], 
    'dibs_mixture_embed_off':       on_off_red[0],  
    'dibs_empirical_embed_on':      on_off_red[1],
    'dibs_mixture_embed_on':        on_off_red[1],

    "dibs_empirical_ndim=5":        sequential_red[0], 
    "dibs_empirical_ndim=7":        sequential_red[1],
    "dibs_empirical_ndim=10":       sequential_red[2],
    "dibs_empirical_ndim=15":       sequential_red[3],
    "dibs_empirical_ndim=20":       sequential_red[4],
    "dibs_empirical_ndim=30":       sequential_red[4],
    "dibs_empirical_ndim=50":       sequential_red[4],

    "dibs_mixture_ndim=5":          sequential_red[0],
    "dibs_mixture_ndim=7":          sequential_red[1],
    "dibs_mixture_ndim=10":         sequential_red[2],
    "dibs_mixture_ndim=15":         sequential_red[3],
    "dibs_mixture_ndim=20":         sequential_red[4],
    "dibs_mixture_ndim=30":         sequential_red[4],
    "dibs_mixture_ndim=50":         sequential_red[4],

    # "dibs_empirical_ndim=5":        sequential_red[0],
    # "dibs_empirical_ndim=7":        sequential_red[0],
    # "dibs_empirical_ndim=10":       sequential_red[0], 
    # "dibs_empirical_ndim=15":       sequential_red[1],
    # "dibs_empirical_ndim=20":       sequential_red[2],
    # "dibs_empirical_ndim=30":       sequential_red[3],
    # "dibs_empirical_ndim=50":       sequential_red[4],

    # "dibs_mixture_ndim=5":          sequential_red[0],
    # "dibs_mixture_ndim=7":          sequential_red[0],
    # "dibs_mixture_ndim=10":         sequential_red[0],
    # "dibs_mixture_ndim=15":         sequential_red[1],
    # "dibs_mixture_ndim=20":         sequential_red[2],
    # "dibs_mixture_ndim=30":         sequential_red[3],
    # "dibs_mixture_ndim=50":         sequential_red[4],

    # "dibs_empirical_ndim=5":        sequential_red7[0], 
    # "dibs_empirical_ndim=7":        sequential_red7[1],
    # "dibs_empirical_ndim=10":       sequential_red7[2],
    # "dibs_empirical_ndim=15":       sequential_red7[3],
    # "dibs_empirical_ndim=20":       sequential_red7[4],
    # "dibs_empirical_ndim=30":       sequential_red7[5],
    # "dibs_empirical_ndim=50":       sequential_red7[6],

    # "dibs_mixture_ndim=5":          sequential_red7[0],
    # "dibs_mixture_ndim=7":          sequential_red7[1],
    # "dibs_mixture_ndim=10":         sequential_red7[2],
    # "dibs_mixture_ndim=15":         sequential_red7[3],
    # "dibs_mixture_ndim=20":         sequential_red7[4],
    # "dibs_mixture_ndim=30":         sequential_red7[5],
    # "dibs_mixture_ndim=50":         sequential_red7[6],

    "dibs_empirical_steps=100":     '#ffffff',  # white
    "dibs_empirical_steps=200":     sequential_red[0],
    "dibs_empirical_steps=400":     sequential_red[1],
    "dibs_empirical_steps=800":     sequential_red[2],
    "dibs_empirical_steps=1600":    sequential_red[3],
    "dibs_empirical_steps=3200":    sequential_red[4],

    "dibs_mixture_steps=100":       '#ffffff',  # white
    "dibs_mixture_steps=200":       sequential_red[0],
    "dibs_mixture_steps=400":       sequential_red[1],
    "dibs_mixture_steps=800":       sequential_red[2],
    "dibs_mixture_steps=1600":      sequential_red[3],
    "dibs_mixture_steps=3200":      sequential_red[4],
}


method_short = {
    'boot_ges':                     r'BGES',
    'boot_pc':                      r'BPC',

    'joint_boot_ges':               r'BGES${}^*$',
    'joint_boot_pc':                r'BPC${}^*$',

    'mcmc_structure':               r'MC${}^3$',
    'mh_joint_mcmc_structure':      r'M-MC${}^3$',
    'gibbs_joint_mcmc_structure':   r'G-MC${}^3$',

    'dibs_empirical':               r'DiBS',
    'dibs_mixture':                 r'DiBS+',
}

method_colors_written = {v: method_colors[k] for k, v in method_format_dict.items()}

column_order_list = [
    'kl_q_p',
    'kl_p_q',
    'mmd_shd_01_sample',
    'mmd_shd_1_sample',
    'mmd_shd_10_sample',
    'l1_edge_belief',
    'expected_shd',
    'expected_shd_cpdag',
    'roc_auc',
    'prc_auc',
    'ave_prec',
    'best_train_neg_log_marginal_likelihood',
    'best_train_neg_log_likelihood',
    'neg_ave_train_marginal_likelihood',
    'neg_ave_train_likelihood',
    'neg_train_log_posterior_predictive',
    'neg_ave_test_log_marginal_likelihood',
    'neg_ave_test_log_likelihood',
    'neg_test_log_posterior_predictive',
    'neg_ave_interv_log_marginal_likelihood',
    'neg_ave_interv_log_likelihood',
    'neg_interv_log_posterior_predictive',
    'n_unique_graphs',
    'n_cyclic_graphs',
    'walltime',
]


hue_order_all = [
    "mcmc_structure",
    "mh_joint_mcmc_structure",
    "gibbs_joint_mcmc_structure",
    "boot_pc",
    "boot_ges",
    "joint_boot_pc",
    "joint_boot_ges",
    "dibs_empirical",
    "dibs_mixture",

    "dibs_empirical_embed_off",
    "dibs_empirical_embed_on",
    "dibs_mixture_embed_off",
    "dibs_mixture_embed_on",

    "dibs_empirical_ndim=5",
    "dibs_empirical_ndim=7",
    "dibs_empirical_ndim=10",
    "dibs_empirical_ndim=15",
    "dibs_empirical_ndim=20",
    "dibs_empirical_ndim=30",
    "dibs_empirical_ndim=50",
    "dibs_mixture_ndim=5",
    "dibs_mixture_ndim=7",
    "dibs_mixture_ndim=10",
    "dibs_mixture_ndim=15",
    "dibs_mixture_ndim=20",
    "dibs_mixture_ndim=30",
    "dibs_mixture_ndim=50",

    "dibs_empirical_steps=100",
    "dibs_empirical_steps=200",
    "dibs_empirical_steps=400",
    "dibs_empirical_steps=800",
    "dibs_empirical_steps=1600",
    "dibs_empirical_steps=3200",
    "dibs_mixture_steps=100",
    "dibs_mixture_steps=200",
    "dibs_mixture_steps=400",
    "dibs_mixture_steps=800",
    "dibs_mixture_steps=1600",
    "dibs_mixture_steps=3200",
]

def _set_default_axis_settings(ax):
    # Hide the right and top spines
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.ticklabel_format(style='plain')
    return


def _get_figsize(figsize):

    if figsize == 'FIG_SIZE_NEURIPS_DOUBLE':
        return FIG_SIZE_NEURIPS_DOUBLE
    elif figsize == 'FIG_SIZE_NEURIPS_TRIPLE':
        return FIG_SIZE_NEURIPS_TRIPLE
    elif figsize == 'FIG_SIZE_NEURIPS_DOUBLE_TALL':
        return FIG_SIZE_NEURIPS_DOUBLE_TALL
    elif figsize == 'FIG_SIZE_NEURIPS_TRIPLE_TALL':
        return FIG_SIZE_NEURIPS_TRIPLE_TALL
    else:
        raise ValueError('Unknown figsize')

def _save_plot(filename_strgs):
    filename = '_'.join(filename_strgs)
    for format in ['pdf', 'png']:
        save_path = os.path.abspath(os.path.join(
            '..', *PLOT_STORE_ROOT, filename + '.' + format
        ))
        plt.savefig(save_path, format=format, facecolor=None,
            dpi=DPI, bbox_inches='tight')

def adjust_limits(kwargs, axs, row):
    ''''
    Adjusts list of axes `axs` limits based on hard coded numbers
    '''

    for j in range(len(axs)):
        ymin, ymax = axs[j].get_ylim()
        try:
            ymin_new, ymax_new = PLOT_ID_LIMS[kwargs.id][row][j]
            if ymin_new is not None:
                ymin = ymin_new
            if ymax_new is not None:
                ymax = ymax_new
        except:
            pass
        axs[j].set_ylim((ymin, ymax))

    return axs


def plot_eval_violinplot_single(filename, df, n_particles, kwargs):
    ''''
    Plots each metric in one panel
    '''
    # set plot parameters
    sns.set_theme(style="ticks", rc=NEURIPS_RCPARAMS)

    # generate plot for each metric
    for i, metric in enumerate(fields_distributionEvaluation_metrics):

        if 'posterior' in metric and not kwargs.with_posterior_predictive:
            continue

        # filter out correct `n_particles`
        if n_particles is not None:
            df = df[df['n_particles'] == n_particles]

        # filter methods that collect values for this method
        df_metric_measured = df[df[metric].notnull()]
        if df_metric_measured.empty:
            continue

        hue_order = [method for method in hue_order_all if method in df.descr.unique()]

        # if active, average rollouts of method first
        if kwargs.individual_rollouts:
            df_metric_measured = df_metric_measured[['descr', metric, 'c', 'r']]
            grouped = df_metric_measured.groupby(['descr', 'c', 'r']).mean()
            df_metric_measured = pd.DataFrame(grouped.to_records())

        else:    
            df_metric_measured = df_metric_measured[['descr', metric, 'c']]
            grouped = df_metric_measured.groupby(['descr', 'c']).mean()
            df_metric_measured = pd.DataFrame(grouped.to_records())
            

        if metric_scale_dict[metric] == 'log':
            df_metric_measured[metric] = np.log10(df_metric_measured[metric])
            
            # Equivalent to matplotlibs symlog
            # https://iopscience.iop.org/article/10.1088/0957-0233/24/2/027001/meta

            # x_ = df_metric_measured[metric]
            # C = 1 / np.log(10)
            # df_metric_measured[metric] = np.sign(x_) * np.log10(1 + np.abs(x_ / C))

        df_metric_measured["Method"] = df_metric_measured["descr"].map(method_format_dict)

        fig, ax = plt.subplots(1, 1, figsize=kwargs.size_single)

        plot_kwargs = dict(
            ax=ax,
            x="Method",
            y=metric,
            hue="Method",
            dodge=False, # violin positions do not change position or width according to hue
            data=df_metric_measured,
            order=[method_format_dict[w] for w in hue_order],
            hue_order=[method_format_dict[w] for w in hue_order],
            palette=method_colors_written,
        )

        if kwargs.violin:
            sns.violinplot(**plot_kwargs)
        elif kwargs.boxen:
            sns.boxenplot(**plot_kwargs)
        else:
            plot_kwargs['fliersize'] = 1
            sns.boxplot(**plot_kwargs)
                    
        # axis         
        ax.set_xticklabels([])
        ax.set_xlabel("")

        ax.set_ylabel(metric_format_dict.get(metric, metric))
        ax.yaxis.grid()

        if metric_scale_dict[metric] == 'log':
            def logy_format_func(value, tick_number):
                return r"$10^{" + str(int(value)) + r"}$"
            ax.yaxis.set_major_formatter(plt.FuncFormatter(logy_format_func))

        # legend
        plt.legend(loc='upper right')

        # save
        filename_opts = [filename, metric]
        if not kwargs.individual_rollouts:
            filename_opts.append('ave')
        if kwargs.violin:
            filename_opts.append('violin')
        if kwargs.boxen:
            filename_opts.append('boxen')

        _save_plot(filename_opts)
        plt.close()

    return


def plot_eval_violinplot_tuple(filename, df, n_particles, kwargs, tuple):
    ''''
    Plots each metric in its own panel
    Violinplot arranged by n_particles
    '''

    # set plot parameters
    sns.set_theme(style="ticks", rc=NEURIPS_RCPARAMS)

    flat_tuple = [elt for l in tuple for elt in l]

    # filter metrics not in triple
    df = df.drop(df.columns.difference(flat_tuple + [
        'descr',
        'n_particles',
        'c',
        'r',
        'graph_dist'
    ]), 1)

    # filter empty metrics (marginal vs joint metrics)
    flat_tuple = [elt for l in tuple for elt in l]
    for col in df.columns:
        if len(df[col].value_counts()) == 0:
            df = df.drop(col, 1)

    # filter methods that collect values for this method
    for metric in df.columns:
        df = df[df[metric].notnull()]

    tuple_metrics = [item for item in df.columns if item in set(flat_tuple)]
    tuple_metrics = [item for item in column_order_list if item in tuple_metrics] # reorder
    print('tuple_metrics', tuple_metrics)

    # filter out correct `n_particles`
    if n_particles is not None:
        df = df[df['n_particles'] == n_particles]

    if df.empty:
        raise ValueError('No method measures any values for any metric in the tuple.')
    
    hue_order = [method for method in hue_order_all if method in df.descr.unique()]

    # legend size adjustment
    if kwargs.ablation_steps:
        # right
        kwargs.size_tuple = (kwargs.size_tuple[0] + 0.2, kwargs.size_tuple[1])
    elif kwargs.ablation_dim:
        # right
        kwargs.size_tuple = (kwargs.size_tuple[0], kwargs.size_tuple[1])
    elif kwargs.ablation_embed or kwargs.tuple_legend:
        # bottom 
        kwargs.size_tuple = (kwargs.size_tuple[0] - 0.3, kwargs.size_tuple[1] + 0.2)

    fig, axs = plt.subplots(1, len(tuple_metrics), figsize=kwargs.size_tuple)

    xticks = []

    print(f"\nCOUNT:")
    df_print__ = df.groupby(by='descr').agg('count')
    print(df_print__['c'])

    for i, metric in enumerate(tuple_metrics):

        df_metric_measured = df.copy()

        # if active, average rollouts of method first
        if kwargs.individual_rollouts:
            df_metric_measured = df_metric_measured[['descr', metric, 'n_particles', 'c', 'r', 'graph_dist']]
            grouped = df_metric_measured.groupby(['descr', 'n_particles', 'c', 'r', 'graph_dist']).mean()
            df_metric_measured = pd.DataFrame(grouped.to_records())

        else:    
            df_metric_measured = df_metric_measured[['descr', metric, 'n_particles', 'c', 'graph_dist']]
            grouped = df_metric_measured.groupby(['descr', 'n_particles', 'c', 'graph_dist']).mean()
            df_metric_measured = pd.DataFrame(grouped.to_records())

        if metric_scale_dict[metric] == 'log':
            df_metric_measured[metric] = np.log10(df_metric_measured[metric])
            
            # Equivalent to matplotlibs symlog
            # https://iopscience.iop.org/article/10.1088/0957-0233/24/2/027001/meta

            # x_ = df_metric_measured[metric]
            # C = 1 / np.log(10)
            # df_metric_measured[metric] = np.sign(x_) * np.log10(1 + np.abs(x_ / C))

        # nice method name 
        df_metric_measured["Method"] = df_metric_measured["descr"].map(method_format_dict)
        df_metric_measured["Graph"] = df_metric_measured["graph_dist"]

        # plot
        plot_kwargs = dict(
            ax=axs[i],
            x="Method",
            y=metric,
            hue="Method",
            dodge=False, # violin positions do not change position or width according to hue
            data=df_metric_measured,
            order=[method_format_dict[w] for w in hue_order],
            hue_order=[method_format_dict[w] for w in hue_order],
            palette=method_colors_written,
            saturation=COLOR_SATURATION,
        )

        if kwargs.violin:
            sns.violinplot(**plot_kwargs)
        elif kwargs.boxen:
            sns.boxenplot(**plot_kwargs)
        else:
            plot_kwargs['fliersize'] = 1
            sns.boxplot(**plot_kwargs)
                    
        # axis      
        axs[i].set_xticklabels([])
        axs[i].set_xlabel("")

        if kwargs.ablation_embed:
            xticks = [
                "DiBS", "",
                "DiBS+", "",
            ]
            axs[i].set_xticklabels(xticks)
            axs[i].set_xlabel("")

            # move xtick labels right to look nicer
            dx = 12/72.; dy = 0/72. 
            offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
            for label in axs[i].xaxis.get_majorticklabels():
                label.set_transform(label.get_transform() + offset)

        elif kwargs.ablation_dim:
            xticks = [
                "", "", "DiBS", "", "",
                "", "", "DiBS+", "", "",
            ]
            axs[i].set_xticklabels(xticks)
            axs[i].set_xlabel(r"$k$")

        elif kwargs.ablation_steps:
            xticks = [
                "", "", "DiBS", "", "",
                "", "", "DiBS+", "", "",
            ]
            axs[i].set_xticklabels(xticks)

        else:
            axs[i].set_xticklabels([method_short.get(w, w) for w in hue_order], rotation=45, fontsize=7, ha='center', va="center", position=(0,-0.15))


        # axs[r, i].set_ylabel(metric_format_dict.get(metric, metric))
        axs[i].set_ylabel("")
        axs[i].yaxis.grid()
        axs[i].set_title(metric_format_dict.get(metric, metric))
        axs[i].yaxis.set_major_locator(plt.MaxNLocator(kwargs.n_yticks))

        if metric_scale_dict[metric] == 'log':
            def logy_format_func(value, tick_number):
                return r"$10^{" + str(int(value)) + r"}$"
            axs[i].yaxis.set_major_formatter(plt.FuncFormatter(logy_format_func))

        # legend
        axs[i].legend([], [], frameon=False)
        handles, labels = axs[i].get_legend_handles_labels()
            
    # multi figure legend
    if kwargs.ablation_embed or kwargs.ablation_dim or kwargs.ablation_steps:
        handles = handles[:len(handles) // 2]
        labels = labels[:len(labels) // 2]
    
    # legend size adjustment
    if kwargs.ablation_steps or kwargs.ablation_dim:
        # right
        axs[-1].legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5), borderaxespad=1.0)
    elif kwargs.ablation_embed or kwargs.tuple_legend:
        # bottom 
        axs[-1].legend(handles, labels, loc='upper right', bbox_to_anchor=(1.07, -0.15), borderaxespad=1.0)


    axs = adjust_limits(kwargs, axs, row='single')

    # tight layout
    fig.tight_layout()

    # save
    filename_opts = [filename, "tuple"]
    if not kwargs.individual_rollouts:
        filename_opts.append('ave')
    if kwargs.violin:
        filename_opts.append('violin')
    if kwargs.boxen:
        filename_opts.append('boxen')

    _save_plot(filename_opts)
    plt.close()
   
    return


def plot_eval_violinplot_tuple_rows(filename, df, n_particles, kwargs, tuple, row_str):

    # set plot parameters
    rc = copy.deepcopy(NEURIPS_RCPARAMS)
    rc["axes.labelpad"] = 5.0
    sns.set_theme(style="ticks", rc=rc)

    flat_tuple = [elt for l in tuple for elt in l]

    # filter metrics not in triple
    cols_necessary = [
        'descr',
        'n_particles',
        'c',
        'r',
        'graph_dist'
    ]
    df = df.drop(df.columns.difference(flat_tuple + cols_necessary), 1)
    assert(row_str in cols_necessary)

    # filter empty metrics (marginal vs joint metrics)
    flat_tuple = [elt for l in tuple for elt in l]
    for col in df.columns:
        if len(df[col].value_counts()) == 0:
            df = df.drop(col, 1)

    # filter methods that collect values for this method
    for metric in df.columns:
        df = df[df[metric].notnull()]

    tuple_metrics = [item for item in df.columns if item in set(flat_tuple)]
    tuple_metrics = [item for item in column_order_list if item in tuple_metrics] # reorder
    print('tuple_metrics', tuple_metrics)

    # filter out correct `n_particles`
    if n_particles is not None:
        df = df[df['n_particles'] == n_particles]

    if df.empty:
        raise ValueError('No method measures any values for any metric in the tuple.')

    # number of rows
    rows = list(df[row_str].unique())
    rows = [row for row in row_order_dict[row_str] if row in rows]

    # plotting
    hue_order = [method for method in hue_order_all if method in df.descr.unique()]
    metric_mins = defaultdict(set)
    metric_maxs = defaultdict(set)
   
    fig, axs = plt.subplots(len(rows), len(tuple_metrics), figsize=kwargs.size_tuple_row)
    # fig, axs = plt.subplots(len(rows), len(tuple_metrics), figsize=kwargs.size_tuple_row, gridspec_kw={'wspace': 0.5, 'hspace': 0.17})

    for r, row_feature in enumerate(rows):
    
        print(f"\nCOUNT: {row_feature}")
        df_print__ = df[df[row_str] == row_feature].groupby(by='descr').agg('count')
        print(df_print__['c'])

        for i, metric in enumerate(tuple_metrics):

            df_metric_measured = df.copy()

            # filter row feature
            df_metric_measured = df_metric_measured[df_metric_measured[row_str] == row_feature]

            # if active, average rollouts of method first
            if kwargs.individual_rollouts:
                df_metric_measured = df_metric_measured[['descr', metric, 'n_particles', 'c', 'r', 'graph_dist']]
                grouped = df_metric_measured.groupby(['descr', 'n_particles', 'c', 'r', 'graph_dist']).mean()
                df_metric_measured = pd.DataFrame(grouped.to_records())

            else:    
                df_metric_measured = df_metric_measured[['descr', metric, 'n_particles', 'c', 'graph_dist']]
                grouped = df_metric_measured.groupby(['descr', 'n_particles', 'c', 'graph_dist']).mean()
                df_metric_measured = pd.DataFrame(grouped.to_records())

            if metric_scale_dict[metric] == 'log':
                df_metric_measured[metric] = np.log10(df_metric_measured[metric])
                
                # Equivalent to matplotlibs symlog
                # https://iopscience.iop.org/article/10.1088/0957-0233/24/2/027001/meta

                # x_ = df_metric_measured[metric]
                # C = 1 / np.log(10)
                # df_metric_measured[metric] = np.sign(x_) * np.log10(1 + np.abs(x_ / C))

            # nice method name 
            df_metric_measured["Method"] = df_metric_measured["descr"].map(method_format_dict)
            df_metric_measured["Graph"] = df_metric_measured["graph_dist"]

            # plot
            plot_kwargs = dict(
                ax=axs[r, i],
                x="Method",
                y=metric,
                hue="Method",
                dodge=False, # violin positions do not change position or width according to hue
                data=df_metric_measured,
                order=[method_format_dict[w] for w in hue_order],
                hue_order=[method_format_dict[w] for w in hue_order],
                palette=method_colors_written,
                saturation=COLOR_SATURATION,
            )

            if kwargs.violin:
                sns.violinplot(**plot_kwargs)
            elif kwargs.boxen:
                sns.boxenplot(**plot_kwargs)
            else:
                plot_kwargs['fliersize'] = 1
                sns.boxplot(**plot_kwargs)
                        
            # axis         
            axs[r, i].set_xticklabels([])
            axs[r, i].set_xlabel("")

            if kwargs.ablation_embed or kwargs.ablation_dim or kwargs.ablation_steps:
                pass     
            else:
                if r == len(rows) - 1:
                    axs[r, i].set_xticklabels([method_short.get(w, w) for w in hue_order], rotation=45, fontsize=7, ha='center', va="center", position=(0,-0.15))
                        
            axs[r, i].yaxis.set_major_locator(plt.MaxNLocator(kwargs.n_yticks))
    
            # axs[r, i].set_ylabel(metric_format_dict.get(metric, metric))
            axs[r, i].set_ylabel("")
            axs[r, i].yaxis.grid()
            ymin, ymax = axs[r, i].get_ylim()
            metric_mins[metric].add(ymin)
            metric_maxs[metric].add(ymax)

            if r == 0:
                axs[r, i].set_title(metric_format_dict.get(metric, metric))
            if i == 0:
                axs[r, i].set_ylabel(row_format_dict.get(row_feature, row_feature))

            if metric_scale_dict[metric] == 'log':
                def logy_format_func(value, tick_number):
                    return r"$10^{" + str(int(value)) + r"}$"
                axs[r, i].yaxis.set_major_formatter(plt.FuncFormatter(logy_format_func))

            elif kwargs.yaxis_likelihood_scientific:
                # only for LL and I-LL
                if i >= 2:
                    # axs[r, i].yaxis.set_major_formatter(plt.FuncFormatter(logy_format_func))
                    # axs[r, i].ticklabel_format(axis='y', style='sci')

                    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
                    g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))#.replace('\\times', '\,\\cdot\,'))
                    axs[r, i].yaxis.set_major_formatter(mticker.FuncFormatter(g))
                    
            # legend
            axs[r, i].legend([], [], frameon=False)
            handles, labels = axs[r, i].get_legend_handles_labels()

    # update y axis limits to have the same for all rows
    # for r, row_feature in enumerate(rows):
    #     for i, metric in enumerate(tuple_metrics):
    #         ymin = min(metric_mins[metric])
    #         ymax = max(metric_maxs[metric])
    #         axs[r, i].set_ylim((ymin, ymax))

    axs[0] = adjust_limits(kwargs, axs[0], row='top')
    axs[1] = adjust_limits(kwargs, axs[1], row='bottom')

    # multi figure legend
    if kwargs.row_legend:
        axs[-1, -1].legend(handles, labels, loc='upper right', bbox_to_anchor=(1.07, -0.35), borderaxespad=1.0)

    # align y labels
    fig.align_ylabels(axs[:, 0])

    # tight layout
    # fig.tight_layout()
    fig.tight_layout(h_pad=0.0, w_pad=0.0)

    # save
    filename_opts = [filename, "tuple", "row-" + row_str]
    if not kwargs.individual_rollouts:
        filename_opts.append('ave')
    if kwargs.violin:
        filename_opts.append('violin')
    if kwargs.boxen:
        filename_opts.append('boxen')

    _save_plot(filename_opts)
    plt.close()
   
    return


def plot_eval_threshold_metrics(filename, df, n_particles, kwargs):
    ''''
    Plots ROC and precision-recall curves
    '''

    # set plot parameters
    sns.set_theme(style="ticks", rc=NEURIPS_RCPARAMS)

    # filter metrics not needed
    df = df.drop(df.columns.difference([
        'descr',
        'n_particles',
        'c',
        'r',
        'graph_dist',
        'fpr',
        'tpr',
        'roc_auc',
        'precision',
        'recall',
    ]), 1)

    # filter methods that collect values for this method
    for metric in df.columns:
        df = df[df[metric].notnull()]

    # filter out correct `n_particles`
    if n_particles is not None:
        df = df[df['n_particles'] == n_particles]

    if df.empty:
        raise ValueError('No method measures any values for any metric in the tuple.')

    hue_order = [method for method in hue_order_all if method in df.descr.unique()]

    # convert list to arrays
    to_jnp_array = lambda lst: jnp.array(ast.literal_eval(lst), dtype=jnp.float64)
    df['fpr'] = df['fpr'].apply(to_jnp_array)
    df['tpr'] = df['tpr'].apply(to_jnp_array)
    df['precision'] = df['precision'].apply(to_jnp_array)
    df['recall'] = df['recall'].apply(to_jnp_array)

    # nice method name
    df["Method"] = df["descr"].map(method_format_dict)

    fig, axs = plt.subplots(1, 2, figsize=kwargs.size_threshold)

    xticks = []

    print(f"\nCOUNT:")
    df_print__ = df.groupby(by='descr').agg('count')
    print(df_print__['c']) 

    # find mean/std metrics for all runs via linear interpolation of curves  
    base = np.linspace(0, 1, 1001)
    interp_tprs, interp_precisions = defaultdict(list), defaultdict(list)
    for method in hue_order:
        method_stats = df.loc[df["descr"] == method]
        for _, row in method_stats.iterrows():
            tprs = np.interp(base, row['fpr'], row['tpr'])
            precs = np.interp(base, row['recall'][::-1], row['precision'][::-1])
            
            # adjust interpolation degeneracies
            if row['tpr'][0] == 0.0:
                tprs[0] = 0.0
            if row['tpr'][-1] == 1.0:
                tprs[-1] = 1.0
            if row['precision'][::-1][0] == 1.0:
                precs[0] = 1.0
            if row['precision'][::-1][-1] == 0.0:
                precs[-1] = 0.0

            interp_tprs[method].append(tprs)
            interp_precisions[method].append(precs)

    # plotting
    for method in hue_order:

        method_stats = df.loc[df["descr"] == method]
        label_ = method_stats['Method'].values[0]
        
        # roc
        mean_tpr = np.array(interp_tprs[method]).mean(0)
        std_tpr = np.array(interp_tprs[method]).std(0)
        axs[0].plot(base, mean_tpr, label=label_, color=method_colors_written[label_])
        if kwargs.threshold_metrics_std:
            axs[0].plot(base, mean_tpr - std_tpr, linestyle='dotted', linewidth=0.7, color=method_colors_written[label_])
            axs[0].plot(base, mean_tpr + std_tpr, linestyle='dotted', linewidth=0.7, color=method_colors_written[label_])

        # precision-recall
        mean_precision = np.array(interp_precisions[method]).mean(0)
        std_precision = np.array(interp_precisions[method]).std(0)
        axs[1].plot(base, mean_precision, label=label_, color=method_colors_written[label_])
        if kwargs.threshold_metrics_std:
            axs[1].plot(base, mean_precision - std_precision, linestyle='dotted', linewidth=0.7, color=method_colors_written[label_])
            axs[1].plot(base, mean_precision + std_precision, linestyle='dotted', linewidth=0.7, color=method_colors_written[label_])

    axs[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate")
    axs[0].set_title('ROC')
    axs[1].set(xlabel="Recall", ylabel="Precision")
    axs[1].set_title('Precision-Recall')

    axs[0].set_xlim((0, 1))
    axs[0].set_ylim((0, 1))
    axs[1].set_xlim((0, 1))
    axs[1].set_ylim((0, 1))

    # legend
    axs[0].legend([], [], frameon=False)
    handles, labels = axs[1].get_legend_handles_labels()
    axs[1].legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5), borderaxespad=1.0)

    # tight layout
    fig.tight_layout()

    # save
    filename_opts = [filename, "threshold"]

    _save_plot(filename_opts)
    plt.close()
   
    return


def _make_plot_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="path to df")

    parser.add_argument("--id", help="id string to adjust ylimits")

    # plot
    parser.add_argument("--single", action="store_true", help="single panel of each metric")
    parser.add_argument("--aggregate_graph_dist", action="store_true", help="avoids splitting violin plots into graph types")

    parser.add_argument("--threshold_metrics", action="store_true", help="pltos ROC and precision-recall curves")
    parser.add_argument("--threshold_metrics_std", action="store_true", help="pltos ROC and precision-recall curves")

    parser.add_argument("--tuple_legend", action="store_true", help="add legend in tuple plot")
    parser.add_argument("--row_legend", action="store_true", help="add legend in row plot")
    parser.add_argument("--violin", action="store_true", help="violin instead of box plot")
    parser.add_argument("--boxen", action="store_true", help="boxen instead of box plot")
    parser.add_argument("--individual_rollouts", action="store_true")
    parser.add_argument("--yaxis_likelihood_scientific", action="store_true")

    parser.add_argument("--with_posterior_predictive", action="store_true")
    parser.add_argument("--n_yticks", default=5)
    parser.add_argument("--ablation_embed", action="store_true", help="ablation mode")
    parser.add_argument("--ablation_dim", action="store_true", help="ablation mode")
    parser.add_argument("--ablation_steps", action="store_true", help="ablation mode")

    # formatting
    parser.add_argument("--size_single", nargs='+', type=float, default=[2.7, 1.5])
    parser.add_argument("--size_tuple", nargs='+', type=float, default=[6.0, 1.5])
    parser.add_argument("--size_tuple_row", nargs='+', type=float, default=[6.0, 2.5])
    parser.add_argument("--size_threshold", nargs='+', type=float, default=[4.5, 1.5])

    return parser


def save_mean_df(df, path, drop=[
    # "l1_edge_belief",
    "expected_shd",
    "best_train_neg_log_marginal_likelihood",
    "best_train_neg_log_likelihood",
    "neg_train_log_posterior_predictive",
    "neg_test_log_posterior_predictive",
    "neg_interv_log_posterior_predictive",
]):
    """
    Saves mean aggregate df over c and r
    """

    # discard unnecessary columns
    df = df.drop(columns=['passed_key', 'status', 'error',
                          'log', 'descr_hparams', 'target_filename', 'n_ho_observations'] + drop)

    # filter empty metrics (marginal vs joint metrics)
    for col in df.columns:
        if len(df[col].value_counts()) == 0:
            df = df.drop(col, 1)

    # aggregate mean and sd
    grouped = df.groupby(['n_vars', 'n_observations', 'graph_dist', 'descr', 'n_particles']).agg([np.mean, np.median, np.std])
    agg_df = pd.DataFrame(grouped.to_records()).drop(columns=["('c', 'mean')", "('c', 'median')", "('c', 'std')", "('r', 'mean')", "('r', 'median')", "('r', 'std')"])

    # round
    agg_df = agg_df.round(4)
    
    # sort correctly
    sorterIndex = dict(zip(hue_order_all, range(len(hue_order_all))))
    agg_df['__sort_rank__'] = agg_df['descr'].map(sorterIndex)
    agg_df.sort_values(['graph_dist', '__sort_rank__'], ascending = [True, True], inplace = True)
    agg_df.drop('__sort_rank__', 1, inplace=True)

    # save
    prefix = 'table_agg_'
    save_path = os.path.abspath(os.path.join(
        '..', *PLOT_STORE_ROOT, prefix + os.path.basename(path) 
    ))

    agg_df.to_csv(save_path)
    print('Saved mean dataframe at: ', save_path)
    return agg_df


def write_latex_table(summary_df, path, marginal_cols=[
    ("expected_shd_cpdag", 1, 1),
    ("roc_auc", 3, 3),
    ("neg_ave_test_log_marginal_likelihood", 1, 1),
    ("neg_ave_interv_log_marginal_likelihood", 1, 1),
], joint_cols=[
    ("expected_shd_cpdag", 1, 1),
    ("roc_auc", 3, 3),
    ("neg_ave_test_log_likelihood", 1, 1),
    ("neg_ave_interv_log_likelihood", 1, 1),
]):
    """
    Write .txt with rows in a latex column, separated by `&`
    """

    # check whether this df contains marginal or joint metrics
    if any([('marginal' in method) for method in summary_df.columns]):
        cols = marginal_cols
    else:
        cols = joint_cols


    file = []
    # print each d/graph combination separately
    for d in summary_df['n_vars'].unique():
        for graph_dist in summary_df['graph_dist'].unique():
            
            file.append(f'd = {d}  graph = {graph_dist}')

            # add new line for each method containing the metrics
            hue_order = [method for method in hue_order_all if method in df.descr.unique()]
            file.append('  '.join([m[0] for m in cols]))
            file.append('')

            for descr in hue_order:

                row_str = f'{descr}  &  '
                row = summary_df[
                    (summary_df['n_vars'] == d) &
                    (summary_df['graph_dist'] == graph_dist) &
                    (summary_df['descr'] == descr)
                ]

                for j, (col, mean_prec, std_prec) in enumerate(cols):
                    try:
                        mean_float = row[f"('{col}', 'mean')"].values.round(mean_prec)[0]
                        stddev_float = row[f"('{col}', 'std')"].values.round(std_prec)[0]
                        row_str += f'{mean_float} $\pm$ {stddev_float}'
                    except:
                        pass
                    if j < len(cols) - 1:
                        row_str += '  &  '
                    else:
                        row_str += '  \\\\'
                    
                file.append(row_str)        

            file.append('')
            file.append('')

    # save
    prefix = 'latex_table_agg_'
    save_path = os.path.abspath(os.path.join(
        '..', *PLOT_STORE_ROOT, prefix + os.path.basename(path) + '.txt'
    ))

    # agg_df.to_csv(save_path)
    with open(save_path, "w") as f:
        for row in file:
            f.write(row + "\n")
        
    print('Saved LaTeX dataframe at: ', save_path)

    return

if __name__ == '__main__':

    parser = _make_plot_parser()
    kwargs = parser.parse_args()

    path = kwargs.path
    df_all = pd.read_csv(path, index_col=0)

    filename = os.path.basename(os.path.normpath(path))
    filename = filename[:-4] # remove .csv

    # extract data from df
    df_gt = df_all.loc[df_all['descr'] == 'gt']
    df = df_all.loc[df_all['descr'] != 'gt']


    n_particles_loop = np.sort(df.n_particles.unique())

    # filter methods that raised an error
    df_error_runs = df.loc[df['status'] == 1]
    error_methods = df_error_runs.descr.unique()
    # df = df[~df['descr'].isin(error_methods)]
    # print(error_methods, 'were filtered out because an error was raised during evaluation.')

    # plot
    if kwargs.threshold_metrics:
        plot_eval_threshold_metrics(filename, df, None, kwargs)
    elif kwargs.single:
        plot_eval_violinplot_single(filename, df, None, kwargs)
    else:
        tuple = [
            ['expected_shd_cpdag'],
            ['roc_auc'],
            ['neg_ave_test_log_marginal_likelihood'], ['neg_ave_test_log_likelihood'],
            ['neg_ave_interv_log_marginal_likelihood'], ['neg_ave_interv_log_likelihood'],
        ]
        if kwargs.aggregate_graph_dist or kwargs.ablation_embed or kwargs.ablation_dim or kwargs.ablation_steps:
            plot_eval_violinplot_tuple(filename, df, None, kwargs, tuple)
        else:
            plot_eval_violinplot_tuple_rows(filename, df, None, kwargs, tuple, 'graph_dist')
    
    # save mean df
    agg_df = save_mean_df(df, kwargs.path)

    # write latex table 
    # write_latex_table(agg_df, kwargs.path)
    

