import os
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision

import collections
import pprint
import tqdm
import scipy

import jax.numpy as jnp
from jax import random

from dibs.graph.distributions import UniformDAGDistributionRejection, LowerTriangularDAGDistribution
from dibs.utils.graph import make_all_dags, get_mat_idx
from dibs.utils.func import particle_empirical

from dibs.kernel.basic import StructuralHammingSquaredExponentialKernel

from dibs.mcmc.structure import StructureMCMC

from dibs.eval.mmd import MaximumMeanDiscrepancy

from dibs.utils.func import pairwise_structural_hamming_distance, pairwise_squared_norm, kullback_leibler_vec


if __name__ == '__main__':

    key = random.PRNGKey(0)
    jnp.set_printoptions(precision=6, suppress=True)

    '''This script compares samples from three distributions over graphs using 
        Maximum Mean Discrepancy with a squared exponential kernel
        
        StructureMCMC from Unifrom  [Ground truth]  (i.e. log prob = const)

        Uniform distribution      (done using rejection sampling, inefficient but easy for now)
        Non-uniform distribution  (pseudo-uniform (but non-uniform) sampling from randomly permuted random lower-triangular adjacency)
    ''' 
    
    n_vars = 5
    n_samples = 500

    n_mc_burnin = 1000
    n_mc_thinning = 50

    sample_dist_a = True
    sample_dist_b = True
    FACTOR = 1.0  # to validate case where adjacency matrix are not {0,1}

    # ground truth target density for uniform distribution
    def unnormalized_log_prob(g):
        return 0.0

    mcmc = StructureMCMC(
        n_vars=n_vars,
        only_non_covered=False)

    key, subk = random.split(key)
    mc_samples = mcmc.sample(
        key=subk,
        n_samples=int(n_samples),
        unnormalized_log_prob=unnormalized_log_prob,
        burnin=n_mc_burnin, thinning=n_mc_thinning)
    mc_samples = mc_samples * FACTOR

    # comparisons
    all_samples = []
    if sample_dist_a:
        dist_a = UniformDAGDistributionRejection(n_vars=n_vars)
        key, *subk = random.split(key, n_samples + 1)
        dist_a_samples = jnp.array([dist_a.sample_G(key=subk[i], return_mat=True) for i in tqdm.tqdm(range(int(n_samples)))])
        dist_a_samples = dist_a_samples * FACTOR
        all_samples.append(('Uniform (Rejection)', dist_a_samples))

    if sample_dist_b:
        dist_b = LowerTriangularDAGDistribution(n_vars=n_vars)
        key, *subk = random.split(key, n_samples + 1)
        dist_b_samples = jnp.array([dist_b.sample_G(key=subk[i], return_mat=True) for i in tqdm.tqdm(range(int(n_samples)))])
        dist_b_samples = dist_b_samples * FACTOR
        all_samples.append(('Overcounting Pseudo-Uniform', dist_b_samples))

    # record MMD as we use more samples from each
    # hs = [None, 0.01, 0.1, 1, 10]
    print_steps = 5

    for desc, samples in all_samples:

        print('Hamming MMD(MCMC uniform || {})'.format(desc))

        k_hamming = StructuralHammingSquaredExponentialKernel()
        metric_hamming = MaximumMeanDiscrepancy(kernel=k_hamming)

        # use bandwidth for all samples (median heuristic)
        # compute median heuristic `sig` over all combined samples
        p_and_q = jnp.concatenate([mc_samples, samples], axis=0)
        h_hamming = k_hamming.compute_median_heuristic(x=p_and_q, y=p_and_q)
        print(f'h: {h_hamming:8.4f}')

        for i in range(print_steps):
            c = int(n_samples // print_steps) * (i + 1)

            # compute empirical distributions
            mc_dist = particle_empirical(mc_samples[:c].astype(jnp.int32))
            dist = particle_empirical(samples[:c].astype(jnp.int32))

            # particle - particle
            mmd_hamming_pp = metric_hamming.squared_mmd(
                p=mc_samples[:c],
                q=samples[:c],
                mmd_h=h_hamming,
                n_vars=n_vars)

            # particle - dist
            mmd_hamming_pd = metric_hamming.squared_mmd(
                p=mc_samples[:c],
                q=dist,
                mmd_h=h_hamming,
                n_vars=n_vars)

            # dist - particle
            mmd_hamming_dp = metric_hamming.squared_mmd(
                p=mc_dist,
                q=samples[:c],
                mmd_h=h_hamming,
                n_vars=n_vars)

            # dist - dist
            mmd_hamming_dd = metric_hamming.squared_mmd(
                p=mc_dist,
                q=dist,
                mmd_h=h_hamming,
                n_vars=n_vars)

            print('After {:8.0f}  :  particle-particle {:10.6f} particle-dist {:10.6f} dist-particle {:10.6f} dist-dist {:10.6f} '.format(
                c, mmd_hamming_pp, mmd_hamming_pd, mmd_hamming_dp, mmd_hamming_dd))

        print()
