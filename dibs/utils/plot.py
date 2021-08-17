import random
import numpy as np
import igraph as ig
from graph import *
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_matrix_comparison(W_true, W_est, W_est_title='Estimated'):
    """Plots two matrices side-by-side"""

    # cmap = plt.cm.get_cmap('OrRd')
    cmap = 'seismic'
    # norm = mpl.colors.Normalize(vmin=-1., vmax=1.)
    norm = mpl.colors.TwoSlopeNorm(vcenter=0.0)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(121)
    ax.matshow(W_true, cmap=cmap, norm=norm)
    ax.set_title('Ground truth')

    ax = fig.add_subplot(122)
    im = ax.matshow(W_est, cmap=cmap, norm=norm)
    ax.set_title(W_est_title)

    plt.show()
