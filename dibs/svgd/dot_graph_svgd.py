


import tqdm
import time
from datetime import datetime

import jax.numpy as jnp
from jax import jit, vmap, random, grad, vjp, jvp, jacrev
from jax.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal
from jax.lax import stop_gradient
from jax.ops import index, index_add, index_update, index_mul
from jax.nn import sigmoid, log_sigmoid
import jax.lax as lax
from jax.experimental import optimizers

from dibs.utils.graph import eltwise_acyclic_constr, acyclic_constr_nograd
from dibs.utils.func import (
    mask_topk, id2bit, bit2id, log_prob_ids, 
    particle_empirical, kullback_leibler_dist, 
    particle_empirical_mixture,
)
from dibs.eval.mmd import MaximumMeanDiscrepancy
from dibs.eval.metrics import neg_log_posterior_predictive, l1_edge_belief, neg_ave_log_marginal_likelihood
from dibs.kernel.basic import StructuralHammingSquaredExponentialKernel


class DotProductGraphSVGD:
    """
        n_vars:                     number of variables in graph
        n_dim:                      size of latent represention
        kernel :                    object satisfying `BasicKernel` signature with differentiable evaluation
        target_log_prior :          differentiable log prior probability using the probabilities of edges
        target_log_prob :           non-differentiable discrete log probability of target distribution
        optimizer:                  dictionary with at least keys `name` and `stepsize`
        alpha:                      inverse temperature parameter schedule of sigmoid; satisfies t -> float
        beta:                       inverse temperature parameter schedule of prior; satisfies t -> float
        gamma:                      scaling parameter schedule inside acyclicity constraint; satisfies t -> float
        tau:                        inverse temperature parameter schedule of gumbel-softmax; satisfies t -> float
        n_grad_mc_samples:          MC samples in gradient estimator
        n_acyclicity_mc_samples:    MC samples in reparameterization of acyclicity constraint
        repulsion_in_prob_space:    whether to apply kernel to particles the probabilities they encode
        clip:                       lower and upper bound for particles across all dimensions
        grad_estimator:             'score' or 'measure-valued'
        score_function_baseline:    weight of addition in score function baseline; == 0.0 corresponds to not using a baseline
        latent_prior_std:           if == -1.0, latent variables have no prior other than acyclicity
                                    otherwise, uses Gaussian prior with mean = 0 and var = `latent_prior_std` ** 2
        constraint_prior_graph...   how acyclicity is defined in p(Z)
                                    - None:   uses matrix of probabilities and doesn't sample
                                    - 'soft': samples graphs using Gumbel-softmax in forward and backward pass
                                    - 'hard': samples graphs using Gumbel-softmax in backward but Bernoulli in forward pass
        graph_embedding_representation     if true, uses inner product embedding graph model
        fix_rotation:               if true, fixes latent representations U, V at row 0 to 1
    """

    def __init__(self, *, n_vars, n_dim, kernel, target_log_prior, target_log_prob, alpha, beta, gamma, tau, n_grad_mc_samples,
                 optimizer, n_acyclicity_mc_samples, grad_estimator='score', score_function_baseline=0.0, clip=None, 
                 repulsion_in_prob_space=False, latent_prior_std=-1.0, fix_rotation="not", constraint_prior_graph_sampling="soft",
                 graph_embedding_representation=True, verbose=False):
        super(DotProductGraphSVGD, self).__init__()

        self.n_vars = n_vars
        self.n_dim = n_dim
        self.kernel = kernel
        self.target_log_prior = target_log_prior
        self.target_log_prob = target_log_prob 
        self.alpha = jit(alpha)
        self.beta = jit(beta)
        self.gamma = jit(gamma)
        self.tau = jit(tau)
        self.n_grad_mc_samples = n_grad_mc_samples
        self.n_acyclicity_mc_samples = n_acyclicity_mc_samples
        self.repulsion_in_prob_space = repulsion_in_prob_space
        self.grad_estimator = grad_estimator
        self.score_function_baseline = score_function_baseline
        self.latent_prior_std = latent_prior_std
        self.constraint_prior_graph_sampling = constraint_prior_graph_sampling
        self.graph_embedding_representation = graph_embedding_representation
        self.clip = clip
        self.fix_rotation = fix_rotation
        self.optimizer = optimizer

        self.verbose = verbose
        self.has_init_core_functions = False
        self.init_core_functions()

    def vec_to_mat(self, x):
        '''Reshapes particle to latent adjacency matrix form
            last dim gets shaped into matrix
            w:    [..., n_vars * n_vars]
            out:  [..., n_vars, n_vars]
        '''
        return x.reshape(*x.shape[:-1], self.n_vars, self.n_vars)

    def mat_to_vec(self, w):
        '''Reshapes latent adjacency matrix form to particle
            last two dims get flattened into vector
            w:    [..., n_vars, n_vars]
            out:  [..., n_vars * n_vars]
        '''
        return w.reshape(*w.shape[:-2], self.n_vars * self.n_vars)

    def particle_to_hard_g(self, x):
        '''Returns g corresponding to alpha = infinity for particles `x`
            x:   [..., n_vars, n_dim, 2]
            out: [..., n_vars, n_vars]
        '''
        if self.graph_embedding_representation:
            u, v = x[..., 0], x[..., 1]
            scores = jnp.einsum('...ik,...jk->...ij', u, v)
        else: 
            scores = x
        g_samples = (scores > 0).astype(jnp.int32)

        # zero diagonal
        g_samples = index_mul(g_samples, index[..., jnp.arange(scores.shape[-1]), jnp.arange(scores.shape[-1])], 0)
        return g_samples

    def particle_to_sampled_log_prob(self, graph_ids, x, log_expectation_per_particle, t):
        '''
        Compute sampling-based SVGD log posterior for `graph_ids` using particles `x`
            
            graph_ids:  [n_graphs]     
            x:          [n_particles, n_vars, n_dim, 2]
            log_expectation_per_particle: [n_particles]
                        computed apriori from samples given the particles
            t:          [1, ]

            out:        [n_graphs]  log probabilities of `graph_ids`

        '''

        n_particles = x.shape[0]
        assert(self.n_vars <= 5)

        '''Compute sampling-based SVGD log posterior for provided graph ids'''
        g = id2bit(graph_ids, self.n_vars)

        # log p(D | G)
        # [batch_size, 1]
        g_log_probs = self.eltwise_log_prob(g)[..., jnp.newaxis]

        # log p(G | W)
        # [batch_size, n_particles]
        g_particle_likelihoods = self.double_eltwise_latent_log_prob(g, x, t)
    
        # [1, n_particles]
        denominator = log_expectation_per_particle[jnp.newaxis]

        # log p(G | D)
        svgd_sampling_log_posterior = logsumexp(g_log_probs + g_particle_likelihoods - denominator, axis=1) - jnp.log(n_particles)

        return svgd_sampling_log_posterior

        

    def init_core_functions(self):
        '''Defines functions needed for SVGD and uses jit'''

        def sig_(x, t):
            '''Sigmoid with parameter `alpha`'''
            return sigmoid(self.alpha(t) * x)
        self.sig = jit(sig_)

        def log_sig_(x, t):
            '''Log sigmoid with parameter `alpha`'''
            return log_sigmoid(self.alpha(t) * x)
        self.log_sig = jit(log_sig_)

        def edge_probs_(x, t):
            '''
            Edge probabilities encoded by latent representation
                x:      [..., d, k, 2]
                out:    [..., d, d]
            '''
            if self.graph_embedding_representation:
                u, v = x[..., 0], x[..., 1]
                scores = jnp.einsum('...ik,...jk->...ij', u, v)
            else:
                scores = x
            
            probs =  self.sig(scores, t)

            # mask diagonal since it is explicitly not modeled
            probs = index_mul(probs, index[..., jnp.arange(probs.shape[-1]), jnp.arange(probs.shape[-1])], 0.0)
            return probs

        self.edge_probs = jit(edge_probs_)

        def edge_log_probs_(x, t):
            '''
            Edge log probabilities encoded by latent representation
                x:      [..., d, k, 2]
                out:    [..., d, d], [..., d, d]
            The returned tuples are log(p) and log(1 - p)
            '''
            if self.graph_embedding_representation:
                u, v = x[..., 0], x[..., 1]
                scores = jnp.einsum('...ik,...jk->...ij', u, v)
            else:
                scores = x

            log_probs, log_probs_neg =  self.log_sig(scores, t), self.log_sig(-scores, t)

            # mask diagonal since it is explicitly not modeled
            # NOTE: this is not technically log(p), but the way `edge_log_probs_` is used, this is correct
            log_probs = index_mul(log_probs, index[..., jnp.arange(log_probs.shape[-1]), jnp.arange(log_probs.shape[-1])], 0.0)
            log_probs_neg = index_mul(log_probs_neg, index[..., jnp.arange(log_probs_neg.shape[-1]), jnp.arange(log_probs_neg.shape[-1])], 0.0)
            return log_probs, log_probs_neg

            
        self.edge_log_probs = jit(edge_log_probs_)

        def sample_g_(p, subk, n_samples):
            '''
            Sample Bernoulli matrix according to matrix of probabilities
                p:   [d, d]
                out: [n_samples, d, d]
            '''
            g_samples = self.vec_to_mat(random.bernoulli(
                subk, p=self.mat_to_vec(p),
                 shape=(n_samples, self.n_vars * self.n_vars))).astype(jnp.int32)

            # mask diagonal since it is explicitly not modeled
            g_samples = index_mul(g_samples, index[..., jnp.arange(p.shape[-1]), jnp.arange(p.shape[-1])], 0)

            return g_samples

        self.sample_g = jit(sample_g_, static_argnums=(2,))
        # [J, d, d], [J, ], [1, ] -> [n_samples, J, d, d]
        self.eltwise_sample_g = jit(vmap(sample_g_, (0, 0, None), 1), static_argnums=(2,))

        def latent_log_prob(single_g, single_x, t):
            '''
            log p(G | U, V)
                single_g:   [d, d]    
                single_x:   [d, k, 2]
                out:        [1,]
            Defined for gradient with respect to `single_x`, i.e. U and V
            '''
            # [d, d], [d, d]
            log_p, log_1_p = self.edge_log_probs(single_x, t)

            # [d, d]
            log_prob_g_ij = single_g * log_p + (1 - single_g) * log_1_p

            # [1,] # diagonal is masked inside `edge_log_probs`
            log_prob_g = jnp.sum(log_prob_g_ij)

            return log_prob_g
        
        # [n_graphs, d, d], [n_particles, d, k, 2] -> [n_graphs]
        self.eltwise_latent_log_prob = jit(vmap(latent_log_prob, (0, None, None), 0))
        # [n_graphs, d, d], [n_particles, d, k, 2] -> [n_graphs, n_particles]
        self.double_eltwise_latent_log_prob = jit(vmap(self.eltwise_latent_log_prob, (None, 0, None), 1))

        # [d, d], [d, k, 2] -> [d, k, 2]
        grad_latent_log_prob = grad(latent_log_prob, 1)

        # [n_graphs, d, d], [d, k, 2] -> [n_graphs, d, k, 2]
        self.eltwise_grad_latent_log_prob = jit(vmap(grad_latent_log_prob, (0, None, None), 0))
        # self.eltwise_grad_latent_log_prob = vmap(grad_latent_log_prob, (0, None, None), 0)

        # [n_graphs, d, d] -> [n_graphs]
        self.eltwise_log_prob = jit(vmap(self.target_log_prob, 0, 0))
        # [n_graphs, n_samples, d, d] -> [n_graphs, n_samples]
        self.double_eltwise_log_prob = jit(vmap(self.eltwise_log_prob, 0, 0))

        '''
        Data likelihood gradient estimators
        Refer to https://arxiv.org/abs/1906.10652
        Implemented are score-function estimator and measure-valued gradients
        for Bernoulli distribution
        '''

        if self.grad_estimator == 'score':
            def grad_latent_likelihood_sf_(single_x, single_sf_baseline, t, subk):
                '''
                Score function estimator for gradient of expectation over p(D | G) w.r.t latent variables 
                Uses same G for both expectations
                    single_x:           [d, k, 2]
                    single_sf_baseline: [1, ]
                    out:                [d, k, 2], [1, ]
                '''
                # [d, d]
                p = self.edge_probs(single_x, t)

                # [n_grad_mc_samples, d, d]
                g_samples = self.sample_g(p, subk, self.n_grad_mc_samples)

                # same MC samples for numerator and denominator
                n_mc_numerator = self.n_grad_mc_samples
                n_mc_denominator = self.n_grad_mc_samples
                latent_dim = self.n_vars * self.n_vars

                # [n_mc_numerator, ] 
                logprobs_numerator = self.eltwise_log_prob(g_samples)
                logprobs_denominator = logprobs_numerator

                # variance_reduction
                logprobs_numerator_adjusted = lax.cond(
                    self.score_function_baseline <= 0.0,
                    lambda _: logprobs_numerator,
                    lambda _: logprobs_numerator - single_sf_baseline,
                    operand=None)

                # [n_vars * n_dim * 2, n_mc_numerator]
                if self.graph_embedding_representation:
                    grad_x = self.eltwise_grad_latent_log_prob(g_samples, single_x, t)\
                        .reshape(self.n_grad_mc_samples, self.n_vars * self.n_dim * 2)\
                        .transpose((1, 0))
                else:
                    grad_x = self.eltwise_grad_latent_log_prob(g_samples, single_x, t)\
                        .reshape(self.n_grad_mc_samples, self.n_vars * self.n_vars)\
                        .transpose((1, 0))

                # stable computation of exp/log/divide
                # [n_vars * n_dim * 2, ]  [n_vars * n_dim * 2, ]
                log_numerator, sign = logsumexp(a=logprobs_numerator_adjusted, b=grad_x, axis=1, return_sign=True)

                # []
                log_denominator = logsumexp(logprobs_denominator, axis=0)

                # [n_vars * n_dim * 2, ]
                stable_sf_grad = sign * jnp.exp(log_numerator - jnp.log(n_mc_numerator) - log_denominator + jnp.log(n_mc_denominator))

                # [n_vars, n_dim, 2]
                if self.graph_embedding_representation:
                    stable_sf_grad_shaped = stable_sf_grad.reshape(self.n_vars, self.n_dim, 2)
                else:
                    stable_sf_grad_shaped = stable_sf_grad.reshape(self.n_vars, self.n_vars)

                # update baseline
                single_sf_baseline = (self.score_function_baseline * logprobs_numerator.mean(0) +
                                 (1 - self.score_function_baseline) * single_sf_baseline)

                return stable_sf_grad_shaped, single_sf_baseline


            self.grad_latent_likelihood = jit(grad_latent_likelihood_sf_)
            # self.grad_latent_likelihood = grad_latent_likelihood_sf_

        else:
            raise ValueError('Unknown gradient estimator. Available are: `score`, `measure-valued`')
    
        self.eltwise_grad_latent_likelihood = jit(vmap(self.grad_latent_likelihood, (0, 0, None, None), (0, 0)))

        '''
        Acyclicity constraint
        '''
        ###
        # forward functions
        ###

        def __particle_to_soft_graph__(x, eps, t):
            """ 
            Gumbel-softmax / discrete distribution using Logistic(0,1) samples `eps`

                x:    [n_vars, n_dim, 2]
                eps:  [n_vars, n_vars]
            
                out:  [n_vars, n_vars]
            """

            if self.graph_embedding_representation:
                scores = jnp.einsum('...ik,...jk->...ij', x[..., 0], x[..., 1])
            else:
                scores = x

            # probs = 1 / (1 + jnp.exp(- alpha * scores))

            # soft reparameterization using gumbel-softmax/concrete distribution
            # sig_expr = self.tau(t) * (jnp.log(u) - jnp.log(1 - u) + self.alpha(t) * scores)
            # eps ~ Logistic(0,1)
            soft_graph = sigmoid(self.tau(t) * (eps + self.alpha(t) * scores))

            # set diagonal to 0 since it is explicitly not modeled
            soft_graph = index_mul(soft_graph, index[..., jnp.arange(soft_graph.shape[-1]), jnp.arange(soft_graph.shape[-1])], 0.0)
            return soft_graph

        def __particle_to_hard_graph__(x, eps, t):
            """ 
            Bernoulli sample of G using probabilities implied by x

                x:    [n_vars, n_dim, 2]
                eps:  [n_vars, n_vars]
            
                out:  [n_vars, n_vars]
            """
            if self.graph_embedding_representation:
                scores = jnp.einsum('...ik,...jk->...ij', x[..., 0], x[..., 1])
            else:
                scores = x
            # probs = 1 / (1 + jnp.exp(- alpha * scores))

            # simply take hard limit of sigmoid in gumbel-softmax/concrete distribution
            hard_graph = ((self.tau(t) * (eps + self.alpha(t) * scores)) > 0.0).astype(jnp.float64)

            # set diagonal to 0 since it is explicitly not modeled
            hard_graph = index_mul(hard_graph, index[..., jnp.arange(hard_graph.shape[-1]), jnp.arange(hard_graph.shape[-1])], 0.0)
            return hard_graph

        ###
        # derivative functions
        ###

        # jacobian: particle -> G
        # [d, d, d, k, 2]
        jac_particle_to_soft_graph__ = jacrev(__particle_to_soft_graph__, 0)

        # gradient: G -> h
        # [d, d]
        grad_acyclic_constr__ = grad(acyclic_constr_nograd, 0)

        ###
        # composite functions
        ###

        if self.constraint_prior_graph_sampling is None:
            #
            # Matrix of probabilities only (using gamma(t) for scaling)
            #
            def __constraint__(x, eps, t):
                ''' 
                Constraint evaluation for a single particle
                Used to define gradient of prior
                    x:      [n_vars, n_dim, 2]

                    eps:    [n_vars, n_vars] 
                        (ignored in this option of `constraint_prior_graph_sampling`)

                    out:    [1, ]
                '''
                # [n_vars, n_vars] # diagonal is masked inside `edge_probs`
                p = self.edge_probs(x, t)

                # scale to avoid vanishing gradient
                p = self.gamma(t) * p

                h = acyclic_constr_nograd(p, self.n_vars)
                return h

            grad_constraint = grad(__constraint__, 0)

        elif self.constraint_prior_graph_sampling == 'soft':
            #
            # Samples graphs using Gumbel-softmax in forward and backward pass
            #

            def __constraint__(single_x, single_eps, t):
                """ 
                Evaluates continuous acyclicity constraint using 
                Gumbel-softmax instead of Bernoulli samples

                    single_x:      [n_vars, n_dim, 2]
                    single_eps:    [n_vars, n_vars]
                
                    out:  [1,]
                """
                G = __particle_to_soft_graph__(single_x, single_eps, t)
                h = acyclic_constr_nograd(G, self.n_vars)
                return h

            def grad_constraint_single_eps_(single_x, single_eps, t):
                '''
                Pathwise derivative of d/dZ h(G(Z, eps))
                    single_x:      [n_vars, n_dim, 2]
                    single_eps:    [n_vars, n_vars]
                
                    out:  [1,]
                '''
            
                # dG(Z)/dZ: [d, d, d, k, 2]
                # since mapping [d, k, 2] -> [d, d]
                jac_G_Z = jac_particle_to_soft_graph__(single_x, single_eps, t)

                # G(Z)
                # [d, d]
                single_G = __particle_to_soft_graph__(single_x, single_eps, t)

                # dh(G)/dG
                grad_h_G = grad_acyclic_constr__(single_G, self.n_vars)

                # pathwise derivative
                # sum over all i,j in G (from h back to Z)
                # [d, k, 2]
                if self.graph_embedding_representation:
                    dZ = jnp.sum(grad_h_G[:, :, None, None, None] * jac_G_Z, axis=(0, 1))
                else:
                    dZ = jnp.sum(grad_h_G[:, :, None, None] * jac_G_Z, axis=(0, 1))
                return dZ

            # [n_vars, n_dim, 2], [mc_samples, n_vars, n_vars] -> [mc_samples, n_vars, n_dim, 2]
            # grad_constraint_batch_eps = jit(vmap(grad_constraint_single_eps_, (None, 0, None), 0))
            # autodiff avoids representing juge jacobian (jacrev) in memory (only possible for `soft`)
            grad_constraint_batch_eps = jit(vmap(grad(__constraint__, 0), (None, 0, None), 0))

            def grad_constraint(x, key, t):
                '''
                Monte Carlo estimator of expectation of constraint gradient
                    x:    [n_vars, n_dim, 2]                
                    key:  [1,]             
                    out:  [n_vars, n_dim, 2] 
                '''
                key, subk = random.split(key)
                eps = random.logistic(key, shape=(self.n_acyclicity_mc_samples, self.n_vars, self.n_vars))

                mc_gradient = grad_constraint_batch_eps(x, eps, t).mean(0)

                return mc_gradient

        elif self.constraint_prior_graph_sampling == 'hard':
            #
            # Samples graphs using Gumbel-softmax in backward but /Bernoulli/ in forward pass
            #
            def __constraint__(single_x, single_eps, t):
                """ 
                Evaluates continuous acyclicity constraint using 
                Gumbel-softmax instead of Bernoulli samples

                    single_x:      [n_vars, n_dim, 2]
                    single_eps:    [n_vars, n_vars]
                
                    out:  [1,]
                """
                G = __particle_to_hard_graph__(single_x, single_eps, t)
                h = acyclic_constr_nograd(G, self.n_vars)
                return h

            def grad_constraint_single_eps_(single_x, single_eps, t):
                '''
                Pathwise derivative of d/dZ h(G(Z, eps))
                    single_x:      [n_vars, n_dim, 2]
                    single_eps:    [n_vars, n_vars]
                
                    out:  [1,]
                '''
            
                # dG(Z)/dZ: [d, d, d, k, 2]
                # since mapping [d, k, 2] -> [d, d]
                jac_G_Z = jac_particle_to_soft_graph__(single_x, single_eps, t)

                # G(Z)
                # [d, d]
                single_G = __particle_to_hard_graph__(single_x, single_eps, t)

                # dh(G)/dG
                grad_h_G = grad_acyclic_constr__(single_G, self.n_vars)

                # pathwise derivative
                # sum over all i,j in G (from h back to Z)
                # [d, k, 2]
                if self.graph_embedding_representation:
                    dZ = jnp.sum(grad_h_G[:, :, None, None, None] * jac_G_Z, axis=(0, 1))
                else:
                    dZ = jnp.sum(grad_h_G[:, :, None, None] * jac_G_Z, axis=(0, 1))
                return dZ

            # [n_vars, n_dim, 2], [mc_samples, n_vars, n_vars] -> [mc_samples, n_vars, n_dim, 2]
            grad_constraint_batch_eps = jit(vmap(grad_constraint_single_eps_, (None, 0, None), 0))

            def grad_constraint(x, key, t):
                '''
                Monte Carlo estimator of expectation of constraint gradient
                    x:    [n_vars, n_dim, 2]                
                    key:  [1,]             
                    out:  [n_vars, n_dim, 2] 
                '''
                key, subk = random.split(key)
                eps = random.logistic(key, shape=(self.n_acyclicity_mc_samples, self.n_vars, self.n_vars))

                mc_gradient = grad_constraint_batch_eps(x, eps, t).mean(0)

                return mc_gradient

        else:
            raise ValueError('Invalid value in `constraint_prior_graph_sampling`')

        #
        # Collect functions; same signature for all options
        #
        
        # [n_vars, n_dim, 2], [n_vars, n_vars] -> 1
        self.constraint = jit(__constraint__)

        # [A, n_vars, n_dim, 2], [B, n_vars, n_vars] -> [A, B]
        self.eltwise_constraint = jit(vmap(vmap(__constraint__, (None, 0, None), 0), (0, None, None), 0))
        
        # [n_particles, n_vars, n_dim, 2], [n_particles,]  -> [n_particles, n_vars, n_dim, 2]
        self.eltwise_grad_constraint = jit(vmap(grad_constraint, (0, 0, None), 0))


        '''
        Latent prior 
        p(Z) = exp(- beta(t) * h(G|Z))

        if `latent_prior_std` > 0, additional factor with
        elementwise zero-mean diagonal gaussian 
        '''

        def target_log_prior_particle(single_x, t):
            '''
            log p(U, V) approx. log p(G)
                single_x:   [d, k, 2]
                out:        [1,]
            '''
            # [d, d] # masking is done inside `edge_probs`
            single_soft_g = self.edge_probs(single_x, t)

            # [1, ]
            return self.target_log_prior(single_soft_g)

        # [d, k, 2], [1,] -> [d, k, 2]
        grad_target_log_prior_particle = jit(grad(target_log_prior_particle, 0))

        # [n_particles, d, k, 2], [1,] -> [n_particles, d, k, 2]
        self.eltwise_grad_target_log_prior_particle = jit(
            vmap(grad_target_log_prior_particle, (0, None), 0))


        if self.latent_prior_std > 0.0:        
            # gaussian and acyclicity
            def eltwise_grad_latent_prior_(x, subkeys, t):
                grad_prior_x = self.eltwise_grad_target_log_prior_particle(x, t)
                grad_constraint_x = self.eltwise_grad_constraint(x, subkeys, t)
                return grad_prior_x \
                       - self.beta(t) * grad_constraint_x \
                       - 2.0 * x / (self.latent_prior_std ** 2.0)
        else:
            # only acyclicity
            def eltwise_grad_latent_prior_(x, subkeys, t):
                grad_prior_x = self.eltwise_grad_target_log_prior_particle(x, t)
                grad_constraint_x = self.eltwise_grad_constraint(x, subkeys, t)
                return grad_prior_x \
                       - self.beta(t) * grad_constraint_x

        # prior p(W)
        self.eltwise_grad_latent_prior = jit(eltwise_grad_latent_prior_)
        # self.eltwise_grad_latent_prior = eltwise_grad_latent_prior_

        '''
        Kernel eval and grad
        '''  
        if self.repulsion_in_prob_space:
            raise NotImplementedError('Make sure the bottom is correct')
            def f_kernel_(a, b, h, t):
                return self.kernel.eval(
                    x=self.edge_probs_(a, t),
                    y=self.edge_probs_(b, t),
                    h=h)
        else:
            def f_kernel_(a, b, h, t):
                return self.kernel.eval(x=a, y=b, h=h)

        self.f_kernel = jit(f_kernel_)
        grad_kernel = jit(grad(self.f_kernel, 0))
        self.eltwise_grad_kernel = jit(vmap(grad_kernel, (0, None, None, None), 0))

        # define single SVGD particle update for jit and vmap
        def x_update(single_x, kxx_for_x, x, grad_log_prob, h, t):
        
            # compute terms in sum
            if self.graph_embedding_representation:
                weighted_gradient_ascent = kxx_for_x[..., None, None, None] * grad_log_prob
            else:
                weighted_gradient_ascent = kxx_for_x[..., None, None] * grad_log_prob

            repulsion = self.eltwise_grad_kernel(x, single_x, h, t)

            # average and negate
            return - (weighted_gradient_ascent + repulsion).mean(axis=0)

        self.parallel_update = jit(vmap(x_update, (0, 1, None, None, None, None), 0))
        # self.parallel_update = vmap(x_update, (0, 1, None, None, None, None), 0)

        self.has_init_core_functions = True


    def sample_particles(self, *, n_steps, init_particles, key,
            eval_metrics=[], tune_metrics=[], metric_every=1, iter0=0, verbose_indication=0):
        """
        Deterministically transforms particles as provided by `init_particles`
        (or heuristically set by N(0, I) of shape (`n_particles`, `n_dim`))
        to minimize KL to target using SVGD
        """
        last_verbose_indication = 1
        t_start = time.time()
        h_is_none = self.kernel.h == -1.0

        # initialize particles
        x = init_particles
        n_particles = x.shape[0]

        if self.graph_embedding_representation:
            if self.fix_rotation == 'parallel':
                x = index_update(x, index[:, 0, :, :], 1.0 * self.latent_prior_std)

            elif self.fix_rotation == 'orthogonal':
                pm_ones = jnp.where(jnp.arange(x.shape[-2]) % 2, -1.0, 1.0).reshape(1, -1)
                x = index_update(x, index[:, 0, :, 0],   pm_ones * self.latent_prior_std)
                x = index_update(x, index[:, 0, :, 1], - pm_ones * self.latent_prior_std)

            elif self.fix_rotation  == 'not':
                pass 

            else:
                raise ValueError('Invalid `fix_rotation` keyword')

        # initialize score function baseline (one for each particle)
        sf_baseline = jnp.zeros(n_particles)

        # jit core functions
        if not self.has_init_core_functions:
            self.init_core_functions()

        # init optimizer
        if self.optimizer['name'] == 'gd':
            opt_init, opt_update, get_params = optimizers.sgd(self.optimizer['stepsize']/ 10.0) # comparable scale for tuning
        elif self.optimizer['name'] == 'momentum':
            opt_init, opt_update, get_params = optimizers.momentum(self.optimizer['stepsize'])
        elif self.optimizer['name'] == 'adagrad':
            opt_init, opt_update, get_params = optimizers.adagrad(self.optimizer['stepsize'])
        elif self.optimizer['name'] == 'adam':
            opt_init, opt_update, get_params = optimizers.adam(self.optimizer['stepsize'])
        elif self.optimizer['name'] == 'rmsprop':
            opt_init, opt_update, get_params = optimizers.rmsprop(self.optimizer['stepsize'])
        else:
            raise ValueError()
        opt_state = opt_init(x)

        '''execute particle updates'''
        it = tqdm.tqdm(range(iter0, n_steps + iter0), desc='SVGD', disable=not self.verbose)
        for t in it:

            x = get_params(opt_state)

            # make sure same bandwith is used for all calls to k(x,x') if the median heuristic is applied
            h = stop_gradient(self.kernel.compute_median_heuristic(
                x=self.edge_probs_(x, t) if self.repulsion_in_prob_space else x,
                y=self.edge_probs_(x, t) if self.repulsion_in_prob_space else x)) \
                if h_is_none else self.kernel.h

            # d/dz log p(D | z)
            key, subk = random.split(key)
            dx_log_likelihood, sf_baseline = self.eltwise_grad_latent_likelihood(x, sf_baseline, t, subk)

            # d/dz log p(z) (acyclicity)
            key, *batch_subk = random.split(key, n_particles + 1)
            dx_log_prior = self.eltwise_grad_latent_prior(x, jnp.array(batch_subk), t)

            dx_log_prob = dx_log_prior + dx_log_likelihood

            # k(x, x)
            kxx = self.f_kernel(x, x, h, t)

            # transformation phi(x)
            phi = self.parallel_update(x, kxx, x, dx_log_prob, h, t)
            if self.graph_embedding_representation:
                if self.fix_rotation != "not":
                    # do not update u_0 and v_0 as they are fixed to 1
                    phi = index_update(phi, index[:, 0, :, :], 0.0)

            # apply transformation
            # normally: `x += stepsize * phi` but for jax.optimizers is `-phi`
            opt_state = opt_update(t, phi, opt_state)

            # check if something went wrong
            if jnp.any(jnp.isnan(phi)):
                if tune_metrics:
                    exit()
                else:
                    print("NaNs discovered:\n")
                    for descr, arr in [
                        ('phi', phi),
                        ('dx_log_prior', dx_log_prior),
                        ('dx_log_likelihood', dx_log_likelihood),
                        ('kxx', kxx),
                    ]:
                        if jnp.any(jnp.isnan(arr)):
                            print(descr, arr.shape)
                            print(arr, flush=True)
                    raise ValueError

            # evaluate
            if not t % metric_every:
                x = get_params(opt_state)
                step = t // metric_every
                params = dict(
                    key=key,
                    step=step,
                    t=t,
                    x=x,
                    phi=phi,
                    alpha=self.alpha(t),
                    beta=self.beta(t),
                    tau=self.tau(t),
                    h=h,
                )

                if eval_metrics:
                    metrics = ' | '.join(
                        f(params) for f in eval_metrics
                    )
                    it.set_description('SVGD | ' + metrics)

                if tune_metrics:
                    for f in tune_metrics:
                        f(params)
            
            # verbose progress
            if verbose_indication > 0:
                if t >= (last_verbose_indication * n_steps // verbose_indication):
                    print(
                        f'DotProductGraphSVGD   {t} / {n_steps} [{(100 * t / n_steps):3.1f} %'  + 
                        f' | {((time.time() - t_start)/60):.0f} min | {datetime.now().strftime("%d/%m %H:%M")}]',
                        flush=True
                    )
                    last_verbose_indication += 1


        # evaluate metrics once more at the end
        if tune_metrics:
            step = t // metric_every
            x = get_params(opt_state)
            params = dict(
                key=key,
                step=step,
                t=n_steps + iter0,
                x=x,
                phi=phi,
                alpha=self.alpha(t),
                beta=self.beta(t),
                tau=self.tau(t),
                h=h,
            )
            for f in tune_metrics:
                f(params)
        
        x_final = get_params(opt_state)
        return x_final


    """
    Helper functions for progress tracking
    """

    def make_metrics(self, eltwise_log_marg_likelihood, x_ho, target, log_posterior=(None, None), mmd_kernel=None, n_mmd_samples=100, 
        n_kl_samples=10, max_entries_per_batch=1e7, make_exhaustive_metrics=False):
        """
        log_posterior :              distribution tuple of target posterior
        eltwise_log_marg_likelihood: [N, d, d] -> [N, ]
        x_ho:                        [N, d] held-out data
        """

        metrics_exhaustive = {}

        # if self.n_vars <= 5:
        #     # need to compute in batches because n_graphs * n_particles can cause OOM
        #     all_graph_ids = jnp.arange(2 ** (self.n_vars * self.n_vars))
        #     n_graphs = all_graph_ids.shape[0]

        '''Initialization / JIT'''
        # if self.n_vars <= 5:
        #     # KL of sampled SVGD posterior
        #     fast_particle_to_sampled_log_prob = jit(self.particle_to_sampled_log_prob)

        #     @jit
        #     def kl_metric_sampling_init(x, subkeys, t):
        #         '''Generate samples for each particle for expectation in denominator
        #             x :         [n_particles, n_vars, n_dim, 2]
        #             subkeys :   [n_particles]
        #             t :         [1, ]

        #             log_expectation_per_particle: [n_particles]
        #         '''

        #         # [n_particles, d, d]
        #         p = self.edge_probs(x, t)

        #         # [n_kl_samples, n_particles, d, d]
        #         g_samples = self.eltwise_sample_g(p, subkeys, n_kl_samples)

        #         # log E_G|W [ p(D | G) ]
        #         # [n_kl_samples, n_particles]
        #         g_samples_log_probs = self.double_eltwise_log_prob(g_samples)

        #         # [n_particles]
        #         log_expectation_per_particle = logsumexp(g_samples_log_probs, axis=0) - jnp.log(n_kl_samples)

        #         return log_expectation_per_particle


        '''Metrics'''
        if make_exhaustive_metrics:

            assert(log_posterior[0] is not None 
               and log_posterior[1] is not None)

            log_posterior = jnp.array(log_posterior[0]), jnp.array(log_posterior[1])

            # def kl_metric_sampling(params):
            #     '''
            #     Computes backward KL for particles during SVGD
            #     if finte = True, log q(x) for q(x) = 0 is treated as 1e-1000 instead of -inf to see progress
            #     '''
            #     x = params['x']
            #     t = params['t']
            #     key = params['key']

            #     n_particles, n_vars, n_dim, _ = x.shape
            #     assert(n_vars <= 5)

            #     # size of batches; n_graphs * n_particles can cause OOM
            #     batch_size = jnp.floor(max_entries_per_batch / n_particles).astype(jnp.int32)
            #     n_iters = jnp.ceil(n_graphs / batch_size).astype(jnp.int32)

            #     # compute denominator expectation
            #     subkeys = random.split(key, n_particles)
            #     log_expectation_per_particle = kl_metric_sampling_init(x, subkeys, t)

            #     # compute log probabilities of sampling-based posterior 
            #     log_posterior_svgd_probs = jnp.hstack([
            #         fast_particle_to_sampled_log_prob(all_graph_ids[batch_size * b : batch_size * (b + 1)], x, log_expectation_per_particle, t) 
            #         for b in range(n_iters)])

            #     log_posterior_svgd = jnp.vstack([all_graph_ids, log_posterior_svgd_probs]).T

            #     kl_q_p = kullback_leibler_dist(log_posterior_svgd, log_posterior, finite=True)
            #     return 'KL(q,p) [sampled]: {:8.04f}'.format(kl_q_p)
            # 
            # metrics_exhaustive['kl_metric_sampling'] = kl_metric_sampling


            def kl_metric_hard(params):
                '''
                Computes backward KL for particles during SVGD
                if finte = True, log q(x) for q(x) = 0 is treated as 1e-1000 instead of -inf to see progress
                '''
                x = params['x']
                hard_g = self.particle_to_hard_g(x)
                empirical = particle_empirical(hard_g)
                kl_q_p = kullback_leibler_dist(empirical, log_posterior, finite=True)
                return 'KL(q,p) [hard]: {:8.04f}'.format(kl_q_p)


            def kl_metric_hard_mixture(params):
                '''
                Computes backward KL for particles during SVGD
                if finte = True, log q(x) for q(x) = 0 is treated as 1e-1000 instead of -inf to see progress
                '''
                x = params['x']
                hard_g = self.particle_to_hard_g(x)
                mixture = particle_empirical_mixture(hard_g, self.eltwise_log_prob)
                kl_q_p = kullback_leibler_dist(mixture, log_posterior, finite=True)
                return 'KL(q,p) [mixt]: {:8.04f}'.format(kl_q_p)

            metrics_exhaustive['kl_metric_hard'] = kl_metric_hard
            metrics_exhaustive['kl_metric_hard_mixture'] = kl_metric_hard_mixture

            if mmd_kernel is not None:
                mmd = MaximumMeanDiscrepancy(kernel=mmd_kernel)
                def mmd_metric_hard(params):
                    '''
                    Computes MMD(p, q) for particles during SVGD
                    using Categorical ground truth samples
                    '''
                    x = params['x']
                    hard_g = self.particle_to_hard_g(x)
                    key = params['key']

                    # ground truth samples
                    key, subk = random.split(key)
                    mmd_g_sample_ids_idx = random.categorical(subk, log_posterior[1], shape=(n_mmd_samples,))
                    mmd_g_sample_ids = log_posterior[0][mmd_g_sample_ids_idx]
                    mmd_g_samples = id2bit(mmd_g_sample_ids, self.n_vars)
                    
                    # MMD
                    squared_mmd = mmd.squared_mmd(p_samples=mmd_g_samples, q_samples=hard_g)
                    return 'MMD(p,q) [hard]: {:8.04f}'.format(squared_mmd)

                metrics_exhaustive['mmd_metric_hard'] = mmd_metric_hard

        def edge_belief_hard(params):
            '''
            Average log likehood on held-out data
            '''
            x = params['x']
            hard_g = self.particle_to_hard_g(x)
            empirical = particle_empirical(hard_g)
            return 'L1 [hard]: {:8.04f}'.format(l1_edge_belief(dist=empirical, g=target.g))

        def edge_belief_mixture(params):
            '''
            Average log likehood on held-out data
            '''
            x = params['x']
            hard_g = self.particle_to_hard_g(x)
            empirical = particle_empirical_mixture(hard_g, self.eltwise_log_prob)
            return 'L1 [mixture]: {:8.04f}'.format(l1_edge_belief(dist=empirical, g=target.g))

        def log_marginal_likelihood_hard(params):
            '''
            Average log likehood on held-out data
            '''
            x = params['x']
            hard_g = self.particle_to_hard_g(x)
            empirical = particle_empirical(hard_g)
            ll = neg_ave_log_marginal_likelihood(dist=empirical, x=x_ho,
                eltwise_log_target=lambda w_, x_: eltwise_log_marg_likelihood(w_, x_))
            return 'neg MLL [hard]: {:8.04f}'.format(ll)

        def log_marginal_likelihood_mixture(params):
            '''
            Average log likehood on held-out data
            '''
            x = params['x']
            hard_g = self.particle_to_hard_g(x)
            empirical = particle_empirical_mixture(hard_g, self.eltwise_log_prob)
            ll = neg_ave_log_marginal_likelihood(dist=empirical, x=x_ho,
                eltwise_log_target=lambda w_, x_: eltwise_log_marg_likelihood(w_, x_))
            return 'neg MLL [mixt]: {:8.04f}'.format(ll)

        def cyclic_graph_count_hard(params):
            '''
            Computes number of graphs that are cyclic amongst hard graphs implied by x
            '''
            x = params['x']
            n_particles = x.shape[0]
            hard_g = self.particle_to_hard_g(x)
            dag_count = (eltwise_acyclic_constr(hard_g, self.n_vars)[0] == 0).sum()
            return 'Cyclic: {:4.0f}'.format(n_particles - dag_count)

        def unique_graph_count_hard(params):
            '''
            Computes number of unique graphs implied by x
            '''
            x = params['x']
            hard_g = self.particle_to_hard_g(x)
            ids = bit2id(hard_g)
            n_unique = len(jnp.unique(ids))
            return 'Unique: {:4.0f}'.format(n_unique)

        def alpha_print(params):
            return 'alpha: {:4.2f}'.format(params['alpha'])

        def beta_print(params):
            return 'beta: {:4.2f}'.format(params['beta'])

        def h_print(params):
            return 'h: {:6.3f}'.format(params['h'])

        def particle_norm_print(params):
            return 'norm(x): {:9.4f}'.format(jnp.linalg.norm(params['x'].flatten()))

        def phi_norm_print(params):
            return 'norm(phi): {:9.6f}'.format(jnp.linalg.norm(params['phi'].flatten()))

        return dict(
            **metrics_exhaustive, 
            edge_belief_hard=edge_belief_hard,
            edge_belief_mixture=edge_belief_mixture,
            log_marginal_likelihood_hard=log_marginal_likelihood_hard,
            log_marginal_likelihood_mixture=log_marginal_likelihood_mixture,
            cyclic_graph_count_hard=cyclic_graph_count_hard,
            unique_graph_count_hard=unique_graph_count_hard,
            alpha_print=alpha_print,
            beta_print=beta_print,
            h_print=h_print,
            particle_norm_print=particle_norm_print,
            phi_norm_print=phi_norm_print,
        )
