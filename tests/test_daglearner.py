import os
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision

import pandas as pd
import time
import collections
import pprint
import tqdm
import scipy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import jax.numpy as jnp
from jax import random

from dibs.models.dirichletCategorical import BDeu
from dibs.models.linearGaussianGaussian import LinearGaussianGaussian

from dibs.bootstrap.learners import GES, PC

from dibs.graph.distributions import UniformDAGDistributionRejection
from dibs.utils.graph import *
from dibs.utils.func import id2bit, bit2id

from cdt.metrics import (
    get_CPDAG,
    SHD, SHD_CPDAG,
    SID, SID_CPDAG,
    precision_recall
)

# from cdt.data import load_dataset

"""
### Real Datasets from CDT library ###

'''
`sachs` -- Dataset of flow cytometry, real data,
11 variables x 7466
samples; Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., & Nolan,
G. P. (2005). Causal protein-signaling networks derived from
multiparameter single-cell data. Science, 308(5721), 523-529.
''' 
data, gt_graph = load_dataset("sachs")
# discard annotations
data = pd.DataFrame(data=data.to_numpy())
gt_adj = nx_adjacency(gt_graph)
n_vars = data.shape[1]


'''
`dream4` -- multifactorial artificial data of the challenge.
Data generated with GeneNetWeaver 2.0, 5 graphs of 100 variables x 100
samples. Marbach D, Prill RJ, Schaffter T, Mattiussi C, Floreano D,
and Stolovitzky G. Revealing strengths and weaknesses of methods for
gene network inference. PNAS, 107(14):6286-6291, 2010.
'''

data, gr_graph = load_dataset("dream4-1")

# discard annotations
data = pd.DataFrame(data=data.to_numpy())
gt_adj = nx_adjacency(gt_graph)
n_vars = data.shape[1]

"""


if __name__ == '__main__':

    key = random.PRNGKey(0)
    np.set_printoptions(precision=6, suppress=True)
    
    test_runs = 3

    '''
    This script tests GES and PC algorithm using R-based CDT package
    '''

    n_vars = 4
    n_observations = 1000
    g_dist = UniformDAGDistributionRejection(n_vars=n_vars)

    # This closely corresponds to the BIC score 
    # (same Gaussain linear SEM assumption)
    model = LinearGaussianGaussian(
        g_dist=g_dist,
        obs_noise=1.0,
        mean_edge=0.0,
        sig_edge=5.0,
    )

    # G ~ p(G)
    # g = g_dist.sample_G()
    g = ig.Graph(directed=True)
    g.add_vertices(4)
    g.add_edges([(1, 0), (1, 2), (0, 2), (3, 2)])

    # theta ~ p(theta | G)
    key, subk = random.split(key)
    theta = model.sample_parameters(key=subk, g=g)

    # x_1, ..., x_n ~ p(x | theta, G) [n_samples, n_vars]
    key, subk = random.split(key)
    x = model.sample_obs(key=subk, n_samples=n_observations, g=g, theta=theta)
    gt_adj = np.array(g.get_adjacency().data)

    print('Ground truth DAG:     ', adjmat_to_str(gt_adj))
    print('Ground truth CPDAG:   ', adjmat_to_str(get_CPDAG(gt_adj)))

    dag_counter = collections.Counter()
    cpdag_counter = collections.Counter()

    for t in tqdm.tqdm(range(test_runs)):

        # select algorithm
        alg = PC() 
        # alg = GES() # BIC score

        # run CPDAG learning algorithm
        pred_cpdag = alg.learn_cpdag(x=x)

        # generate random extension of CPDAG (i.e. a DAG in the MEC implied by CPDAG)
        key, subk = random.split(key)
        pred_cpdag_oriented = random_consistent_expansion(key=subk, cpdag=pred_cpdag)

        # recover CPDAG
        rec_cpdag = get_CPDAG(np.array(pred_cpdag_oriented))

        # CPDAG recovery of consistent DAG extensions is the same CPDAG
        # assert(np.abs(pred_cpdag_mat - rec_cpdag_mat).sum() == 0)
        assert(np.allclose(pred_cpdag, rec_cpdag))

        # CPDAG recovery of CPDAG is the same CPDAG
        # assert(np.abs(pred_cpdag_mat - get_CPDAG(pred_cpdag_mat)).sum() == 0)
        assert(np.allclose(pred_cpdag, get_CPDAG(np.array(pred_cpdag))))

        dag_counter[tuple(bit2id(pred_cpdag_oriented[jnp.newaxis])[0].tolist())] += 1
        cpdag_counter[tuple(bit2id(pred_cpdag[jnp.newaxis])[0].tolist())] += 1

    # visualize
    print('\nFound {} CPDAGs with frequencies: '.format(sum(cpdag_counter.values())))
    for id, freq in cpdag_counter.most_common():
        cpdag = id2bit(jnp.array([id], dtype=jnp.uint8), n_vars)
        print(freq, adjmat_to_str(cpdag.squeeze(0)))

    print('\nFound {} DAGs with frequencies: '.format(sum(dag_counter.values())))
    for id, freq in dag_counter.most_common():
        dag = id2bit(jnp.array([id], dtype=jnp.uint8), n_vars)
        print(freq, adjmat_to_str(dag.squeeze(0)))

    # # 
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    # gt_graph = nx.DiGraph(gt_adj)
    # pos = nx.planar_layout(gt_graph)
    # for j, g in enumerate([gt_graph, nx.DiGraph(pred_cpdag_mat_oriented)]):
    #     nx.draw(g, pos=pos, with_labels=True, font_size=8, ax=ax[j], labels={i:i for i in range(N)})
    #     ax[j].set_axis_off()
    # plt.show()
