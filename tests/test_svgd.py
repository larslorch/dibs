import os
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision

import collections
import pprint
import tqdm
import numpy as onp

import matplotlib

import matplotlib.pyplot as plt
from matplotlib import animation

import jax.numpy as jnp
from jax import grad, jit, vmap, vjp, jvp
from jax import random

from dibs.svgd.svgd import * 
from dibs.kernel.basic import *
from dibs.eval.mmd import MaximumMeanDiscrepancy

PLOT_STORE_ROOT = ['plots']

def kde_plot(*, ax, data, xlim=(-3, 3), ylim=(-3, 3)):
    '''
    data: [N, 2]
    '''
    xmin, xmax = xlim
    ymin, ymax = ylim
    xx, yy = onp.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = onp.vstack([xx.ravel(), yy.ravel()])
    kernel = stats.gaussian_kde(data.T)
    f = np.reshape(kernel(positions).T, xx.shape)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    cfset = ax.contourf(xx, yy, f, cmap='Blues')
    return ax


def density_plot(*, ax, density, xlim=(-3, 3), ylim=(-3, 3)):
    '''
    density: [:, 2] -> [:]
    '''
    xmin, xmax = xlim
    ymin, ymax = ylim
    xx, yy = onp.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = onp.vstack([xx.ravel(), yy.ravel()]).T
    f = density(positions).reshape(xx.shape)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    cfset = ax.contourf(xx, yy, f, cmap='Blues')
    return ax


if __name__ == '__main__':

    '''
    This script tests SVGD for toy distributions
    '''

    # Generate key which is used to generate random numbers
    key = random.PRNGKey(0)

    target = ExampleTarget2D(key=key)
    kernel = FrobeniusSquaredExponentialKernel()  # h not specified indicates median trick

    # svgd
    stepsize = 0.5
    n_steps = 200
    n_particles = 100
    n_dim = 2
    verbose = True
    animate = True
    
    init_particles = random.normal(key, shape=(n_particles, 2)) + jnp.array([-2.0, 2.0])
    # nx, ny = round(jnp.sqrt(n_particles).item()), round(jnp.sqrt(n_particles).item())
    # x = jnp.linspace(-3, -2, nx)
    # y = jnp.linspace(2, 3, ny)
    # init_particles = jnp.stack(jnp.meshgrid(x, y)).reshape(2, -1).T

    svgd = SVGD(kernel=kernel, target_log_prob=target.log_prob, key=key, verbose=verbose and not animate)

    if not animate:
        # MMD as evaluation metric using GT samples
        gt_samples = target.sample(n_particles) 
        k_mmd = FrobeniusSquaredExponentialKernel()
        discrepancy = MaximumMeanDiscrepancy(kernel=k_mmd)
        h_mmd = k_mmd.compute_median_heuristic(x=gt_samples, y=gt_samples)

        @jit
        def mmd(x):
            return discrepancy.squared_mmd(
                p_samples=gt_samples,
                q_samples=x,
                mmd_h=h_mmd)

        # evaluate
        particles = svgd.sample_particles(n_steps=n_steps, init_particles=init_particles.copy(), stepsize=stepsize, eval_metric=mmd)

    else:        
        # prepare animation
        fig = plt.figure()
        ax = plt.axes(xlim=(-5, 5), ylim=(-5, 5))
        ax = density_plot(ax=ax, density=lambda x: jnp.exp(target.log_prob(x)), xlim=(-5, 5), ylim=(-5, 5))
        svgd.init_core_functions()
        svgd.animate_buffer = init_particles
        scat = plt.scatter(init_particles[:, 0], init_particles[:, 1], color='red', marker='+')
        def step(i):
            particles = svgd.sample_particles(n_steps=1, init_particles=svgd.animate_buffer, stepsize=stepsize)
            scat.set_offsets(particles)
            svgd.animate_buffer = particles

            # save as png sequence
            foldername = "svgd-toy-animation"
            save_path = os.path.abspath(os.path.join(
                '..', *PLOT_STORE_ROOT, foldername, f"frame-{i}.png",
            ))
            plt.savefig(save_path, dpi=100)

            return scat

        anim = animation.FuncAnimation(fig, step, repeat=False,
            frames=n_steps, interval=1)
        plt.show()
