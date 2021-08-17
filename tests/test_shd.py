
import collections
import pprint
import tqdm
import pickle
import igraph as ig
import numpy as onp
from tabulate import tabulate
from varname import nameof
import time

import jax.scipy.stats as jstats
import jax.scipy as jsp
import jax.numpy as jnp
from jax import grad, jit, vmap, vjp, jvp
from jax import random
import jax.nn as nn
from jax.scipy.special import logsumexp
from jax.ops import index, index_add, index_mul, index_update
import jax.lax as lax
from jax.tree_util import tree_flatten, tree_unflatten, tree_map

from dibs.utils.func import mask_topk, id2bit, bit2id, log_prob_ids, particle_joint_empirical
from dibs.utils.tree import tree_index
from dibs.eval.metrics import expected_shd

import numpy as onp

from dibs.eval.target import make_ground_truth_posterior

from cdt.metrics import get_CPDAG, SHD

import numpy as onp

if __name__ == '__main__':

    gt = jnp.array([
        [0, 0, 0, 0],
        [1, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
    ])

    print('gt')
    print(gt)
    print('gt cpdag')
    print(get_CPDAG(onp.array(gt)))


    g1 = jnp.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
    ])

    g2 = jnp.array([
        [0, 0, 0, 0],
        [1, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ])

    g3 = jnp.array([
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
    ])

    print('single SHD')
    for g in [g1, g2, g3]:
        dist = (bit2id(g[None]), jnp.array([0.0]))
        print('SHD cpdag only', expected_shd(dist=dist, g=gt, use_cpdag=True))
        print('SHD simple', expected_shd(dist=dist, g=gt, use_cpdag=False))

        print('matches cdt.SHD cpdag only', expected_shd(dist=dist, g=gt, use_cpdag=True) == SHD(get_CPDAG(onp.array(g)), get_CPDAG(onp.array(gt)), double_for_anticausal=False))
        print('matches cdt.SHD simple    ', expected_shd(dist=dist, g=gt, use_cpdag=False) ==  SHD(onp.array(g), onp.array(gt), double_for_anticausal=False))

        print()

    print('expected SHD')
    for g, g_ in [(gt, gt), (g1, g2), (g2, g3)]:
        gs = jnp.array([g, g_])

        dist = (bit2id(gs), jnp.array([0.0, 0.0]))
        print('SHD cpdag only', expected_shd(dist=dist, g=gt, use_cpdag=True))
        print('SHD simple', expected_shd(dist=dist, g=gt, use_cpdag=False))
        print()
