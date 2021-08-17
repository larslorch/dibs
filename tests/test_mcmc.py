import os
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision

import collections
import pprint 
import tqdm

import jax.numpy as jnp
from jax import random
from jax.scipy.special import logsumexp
from jax.ops import index, index_add, index_mul, index_update

from dibs.graph.distributions import UniformDAGDistributionRejection, LowerTriangularDAGDistribution
from dibs.utils.graph import *
from dibs.utils.func import id2bit, bit2id, particle_empirical

from dibs.models.dirichletCategorical import BDeu
from dibs.models.linearGaussianGaussianEquivalent import BGe
from dibs.models.linearGaussianGaussian import LinearGaussianGaussian

from dibs.mcmc.structure import StructureMCMC

from dibs.utils.func import kullback_leibler_dist


if __name__ == '__main__':

    key = random.PRNGKey(0)
    jnp.set_printoptions(precision=6, suppress=True)

    '''This script tests that the empirical (Dirac) distribution of 
    MCMC samples is able to converge to the closed form density for small p(G | D)
    '''

    # MCMC settings
    burnin = 1e2
    thinning = 20
    n_samples = 30
    only_non_covered = False

    '''Generative model (generates ground truth observations)'''
    n_vars = 20
    # n_vars = 4
    # n_vars = 5

    n_observations = 20

    # gt_g_prior = UniformDAGDistributionRejection(n_vars=n_vars)
    gt_g_prior = LowerTriangularDAGDistribution(n_vars=n_vars)
    
    gt_model = LinearGaussianGaussian(
        g_dist=gt_g_prior,
        mean_edge=0.0,
        sig_edge=1.0,
        obs_noise=0.1,
    )

    # sample data
    key, subk = random.split(key)
    g_gt = gt_g_prior.sample_G(key=subk)

    key, subk = random.split(key)
    theta_gt = gt_model.sample_parameters(key=subk, g=g_gt)

    key, subk = random.split(key)
    x = gt_model.sample_obs(key=subk, n_samples=n_observations, g=g_gt, theta=theta_gt)

    '''Define unnormalized target distribution, i.e. unnormalized posterior
        (assume no model misspecification)

        log `p(G) + log p(D | G), where `p(G)/z = p(G)
    '''

    # g_prior = UniformDAGDistributionRejection(n_vars=n_vars)
    g_prior = LowerTriangularDAGDistribution(n_vars=n_vars)

    model = BGe(
        g_dist=g_prior,
        mean_obs=jnp.zeros(n_vars),
        alpha_mu=1.0, 
        alpha_lambd=n_vars + 2,
    )
    # model = LinearGaussianGaussian(
    #     g_dist=g_prior,
    #     mean_edge=0.0,
    #     sig_edge=1.0,
    #     obs_noise=0.1,
    # )
    
    def unnormalized_log_target(g): 
        return model.g_dist.unnormalized_log_prob(g=g) + \
               model.log_marginal_likelihood_given_g(g=g, x=x)

    def unnormalized_log_prob_single(g, j):
        return model.g_dist.unnormalized_log_prob_single(g=g, j=j) + \
               model.log_marginal_likelihood_given_g_single(g=g, x=x, j=j)


    '''Compute normalized posterior (via exhaustive search) for validation'''
    if n_vars <= 5:

        all_dags = make_all_dags(n_vars=n_vars, return_matrices=False)
        z_g = gt_model.g_dist.log_normalization_constant(all_g=all_dags)
        log_marginal_likelihood = gt_model.log_marginal_likelihood(x=x, all_g=all_dags, z_g=z_g)
        
        posterior_ids, posterior_log_probs = [], []  # (unique graph ids, log probs)
        for j, g_ in enumerate(all_dags):
            binary_mat = jnp.array(graph_to_mat(g_))[jnp.newaxis] # [1, d, d]

            id = bit2id(binary_mat).squeeze(0)
            log_prob = gt_model.log_posterior_graph_given_obs(
                    g=g_, x=x, log_marginal_likelihood=log_marginal_likelihood, z_g=z_g)
            
            posterior_ids.append(id)
            posterior_log_probs.append(log_prob)

        posterior_ids = jnp.array(posterior_ids)
        posterior_log_probs = jnp.array(posterior_log_probs)
        log_posterior = (posterior_ids, posterior_log_probs)

        assert(jnp.allclose(logsumexp(posterior_log_probs), 0, atol=1e-6)) # check valid distribution

    '''Run MCMC'''
    mcmc = StructureMCMC(
        n_vars=n_vars,
        only_non_covered=only_non_covered)

    key, subk = random.split(key)
    samples = mcmc.sample(key=subk, n_samples=n_samples, unnormalized_log_prob=unnormalized_log_target,
                          unnormalized_log_prob_single=unnormalized_log_prob_single,
                          burnin=burnin, thinning=thinning)
    
    '''Validation'''

    # assert that KL of empirical dist to ground truth decreases as we use more MCMC samples
    # only possible since ground truth density p(x) is available
    if n_vars <= 5:
        print()
        print('KL(empirical || ground truth)')

        print_steps = 10
        empirical = jnp.zeros(all_dags.shape[0])
        kls = []

        for cutoff in jnp.linspace(n_samples/print_steps, n_samples, print_steps, endpoint=True).astype(jnp.int32):
            
            empirical_dist = particle_empirical(samples[:cutoff])
            kl = kullback_leibler_dist(p=empirical_dist, q=log_posterior, finite=True).item()
            kls.append(kl)
            print('After {:8.0f}  : {: .6f}'.format(cutoff, kl))

        print('MCMC         ', jnp.exp(empirical_dist[1]))
