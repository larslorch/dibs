


import tqdm
import time

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
from dibs.eval.metrics import neg_log_posterior_predictive, l1_edge_belief, neg_ave_log_marginal_likelihood

from dibs.eval.mmd import MaximumMeanDiscrepancy
from dibs.kernel.basic import StructuralHammingSquaredExponentialKernel
from dibs.exceptions import SVGDNaNError


class BatchedDotProductGraphSVGD:
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
        constraint_prior_graph...   how acyclic is defined in p(Z)
                                    - None:   uses matrix of probabilities and doesn't sample
                                    - 'soft': samples graphs using Gumbel-softmax in forward and backward pass
                                    - 'hard': samples graphs using Gumbel-softmax in backward but Bernoulli in forward pass
        graph_embedding_representation     if true, uses inner product embedding graph model
        fix_rotation:               if true, fixes latent representations U, V at row 0 to 1
    """

    def __init__(self, *, n_vars, n_dim, kernel, target_log_prior, target_log_prob, alpha, beta, gamma, tau, n_grad_mc_samples, 
                 n_acyclicity_mc_samples, optimizer, grad_estimator='score', score_function_baseline=0.0, 
                 clip=None, repulsion_in_prob_space=False, latent_prior_std=-1.0, fix_rotation="not",
                 constraint_prior_graph_sampling=None, graph_embedding_representation=True, verbose=False):
        super(BatchedDotProductGraphSVGD, self).__init__()

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

        grad_latent_log_prob = grad(latent_log_prob, 1)
        # [J, d, d], [d, k, 2] -> [J, d, k, 2]
        self.eltwise_grad_latent_log_prob = jit(vmap(grad_latent_log_prob, (0, None, None), 0))
        # self.eltwise_grad_latent_log_prob = vmap(grad_latent_log_prob, (0, None, None), 0)

        self.eltwise_log_prob = jit(vmap(self.target_log_prob, (0, None), 0))
        # self.eltwise_log_prob = lambda x: (
        #     jnp.array([
        #         self.target_log_prob(x[i]) for i in range(x.shape[0])
        #     ])
        # )

        '''
        Data likelihood gradient estimators
        Refer to https://arxiv.org/abs/1906.10652
        Implemented are score-function estimator and measure-valued gradients
        for Bernoulli distribution
        '''

        if self.grad_estimator == 'score':
            def grad_latent_likelihood_sf_(single_x, single_sf_baseline, t, subk, b):
                '''
                Score function estimator for gradient of expectation over p(D | G) w.r.t latent variables 
                Uses same G for both expectations
                    single_x:           [d, k, 2]
                    single_sf_baseline: [1, ]
                    out:                [d, k, 2]
                    t:                  int -- iteration
                    b:                  int -- batch index
                '''
                # [d, d]
                p = self.edge_probs(single_x, t)

                # [n_grad_mc_samples, d, d]
                g_samples = self.sample_g(p, subk, self.n_grad_mc_samples)

                # same MC samples for numerator and denominator
                n_mc_numerator = self.n_grad_mc_samples
                n_mc_denominator = self.n_grad_mc_samples

                # [n_mc_numerator, ] 
                logprobs_numerator = self.eltwise_log_prob(g_samples, b)
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
    
        self.eltwise_grad_latent_likelihood = jit(vmap(self.grad_latent_likelihood, (0, 0, None, None, None), (0, 0)))
        # self.eltwise_grad_latent_likelihood = lambda x, y, z: (
        #     jnp.array([
        #         self.grad_latent_likelihood(x[i], y, z) for i in range(x.shape[0])
        #     ])
        # )

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

            # mask diagonal since it is explicitly not modeled
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

            # mask diagonal since it is explicitly not modeled
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
            # [d, d] # diagonal is masked inside `edge_probs`
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

            # average
            return (weighted_gradient_ascent + repulsion).mean(axis=0)

        self.parallel_update = jit(vmap(x_update, (0, 1, None, None, None, None), 0))
        # self.parallel_update = vmap(x_update, (0, 1, None, None, None, None), 0)


        # one loop iteration, to be batch processed
        def loop_iter_(x, t, key, b, sf_baseline):
            # make sure same bandwith is used for all calls to k(x,x') if the median heuristic is applied
            h = self.kernel.h

            # d/dz log p(D | z)
            key, subk = random.split(key)
            dx_log_likelihood, sf_baseline = self.eltwise_grad_latent_likelihood(x, sf_baseline, t, subk, b)
            
            # d/dz log p(z) (acyclicity)
            key, *batch_subk = random.split(key, x.shape[0] + 1)
            dx_log_prior = self.eltwise_grad_latent_prior(x, jnp.array(batch_subk), t)

            dx_log_prob = dx_log_prior + dx_log_likelihood

            # k(x, x)
            kxx = self.f_kernel(x, x, h, t)

            # transformation phi(x)
            phi = self.parallel_update(x, kxx, x, dx_log_prob, h, t)
            if self.graph_embedding_representation:
                if self.fix_rotation != "not":
                    # do not update u_0 and v_0 as they are fixed
                    phi = index_update(phi, index[:, 0, :, :], 0.0)
            
            return phi, key, sf_baseline

        self.batch_loop_iter = jit(vmap(loop_iter_, (0, None, 0, 0, 0), (0, 0, 0)))
        # def batch_loop_iter_(w, x, y, z):
        #     a, b = [], []
        #     for i in range(w.shape[0]):
        #         a_, b_ = loop_iter_(w[i], x, y[i], z[i])
        #         a.append(a_)
        #         b.append(b_)
        #     return jnp.array(a), jnp.array(b)
        # self.batch_loop_iter = batch_loop_iter_

        self.has_init_core_functions = True


    def sample_particles(self, *, key, n_steps, init_particles, 
            eval_metrics=[], tune_metrics=[], metric_every=1, iter0=0):
        """
        Deterministically transforms particles as provided by `init_particles`
        """
        self.h_is_none = self.kernel.h == -1.0

        # initialize particles
        batch_x = init_particles
        batch_size = batch_x.shape[0]

        if self.graph_embedding_representation:
            if self.fix_rotation == 'parallel':
                batch_x = index_update(batch_x, index[:, :, 0, :, :], 1.0 * self.latent_prior_std)

            elif self.fix_rotation == 'orthogonal':
                pm_ones = jnp.where(jnp.arange(batch_x.shape[-2]) % 2, -1.0, 1.0).reshape(1, 1, -1)
                batch_x = index_update(batch_x, index[:, :, 0, :, 0],   pm_ones * self.latent_prior_std)
                batch_x = index_update(batch_x, index[:, :, 0, :, 1], - pm_ones * self.latent_prior_std)

            elif self.fix_rotation  == 'not':
                pass 

            else:
                raise ValueError('Invalid `fix_rotation` keyword')


        # initialize score function baseline (one for each particle)
        batch_sf_baseline = jnp.zeros(batch_x.shape[0:2])
        
        # jit core functions
        if not self.has_init_core_functions:
            self.init_core_functions()

        # init optimizer
        if self.optimizer['name'] == 'gd':
            opt_init, opt_update, get_params = optimizers.sgd(self.optimizer['stepsize']/ 10.0) # comparable scale for tuning
        elif self.optimizer['name'] == 'momentum':
            opt_init, opt_update, get_params = optimizers.momentum(self.optimizer['stepsize'], 0.9)
        elif self.optimizer['name'] == 'adagrad':
            opt_init, opt_update, get_params = optimizers.adagrad(self.optimizer['stepsize'])
        elif self.optimizer['name'] == 'adam':
            opt_init, opt_update, get_params = optimizers.adam(self.optimizer['stepsize'])
        elif self.optimizer['name'] == 'rmsprop':
            opt_init, opt_update, get_params = optimizers.rmsprop(self.optimizer['stepsize'])
        else:
            raise ValueError()

        batch_opt_init = jit(vmap(opt_init, 0, 0))
        batch_opt_update = jit(vmap(opt_update, (None, 0, 0), 0))
        batch_get_params = jit(vmap(get_params, 0, 0))

        batch_opt_state = batch_opt_init(batch_x)
        batch_keys = random.split(key, batch_size)

        '''execute particle updates'''
        it = tqdm.tqdm(range(iter0, n_steps + iter0), desc='SVGD', disable=not self.verbose)
        for t in it:
            
            # loop iteration in batch
            batch_ids = jnp.arange(batch_size)
            batch_x = batch_get_params(batch_opt_state)
            batch_phi, batch_keys, batch_sf_baseline = self.batch_loop_iter(batch_x, t, batch_keys, batch_ids, batch_sf_baseline)
            batch_opt_state = batch_opt_update(t, -batch_phi, batch_opt_state)

            # check if something went wrong
            if jnp.any(jnp.isnan(batch_phi)):
                raise SVGDNaNError()
                exit()

            # evaluate
            if not t % metric_every:
                batch_x = batch_get_params(batch_opt_state)
                step = t // metric_every
                params = dict(
                    batch_keys=batch_keys,
                    step=step,
                    t=t,
                    batch_x=batch_x,
                    batch_phi=batch_phi,
                    alpha=self.alpha(t),
                    beta=self.beta(t),
                    gamma=self.gamma(t),
                    tau=self.tau(t),
                )

                if eval_metrics:
                    metrics = ' | '.join(
                        f(params) for f in eval_metrics
                    )
                    it.set_description('SVGD | ' + metrics)

                if tune_metrics:
                    for f in tune_metrics:
                        f(params)

        # evaluate metrics once more at the end
        if tune_metrics:
            step = t // metric_every
            batch_x = batch_get_params(batch_opt_state)
            params = dict(
                batch_keys=batch_keys,
                step=step,
                t=n_steps + iter0,
                batch_x=batch_x,
                batch_phi=batch_phi,
                alpha=self.alpha(t),
                beta=self.beta(t),
                gamma=self.gamma(t),
                tau=self.tau(t),
            )
            for f in tune_metrics:
                f(params)
        
        batch_x_final = batch_get_params(batch_opt_state)
        return batch_x_final

    def make_metrics(self, variants_target, eltwise_log_marg_likelihood):

        n_variants = len(variants_target)

        def neg_ave_log_marginal_likelihood_mixture(params):
            '''
            Computes negative log posterior predictive on held-out data
            '''
           
            batch_particles = params['batch_x']
            batch_size = batch_particles.shape[0]
            batch_x_ho = [jnp.array(variants_target[b % n_variants].x_ho) for b in range(batch_size)]

            batch_hard_g = self.particle_to_hard_g(batch_particles)
            batch_mixture = [
                particle_empirical_mixture(
                    hard_g, lambda g: self.eltwise_log_prob(g, b))
                for b, hard_g in zip(jnp.arange(batch_size), batch_hard_g)
            ]  

            # average log posterior predictive on held-out data
            mean_neg_ave_log_marginal_likelihood = jnp.array([
                neg_ave_log_marginal_likelihood(dist=q_, x=x_ho_,
                    eltwise_log_target=lambda w_, x_: eltwise_log_marg_likelihood(w_, x_))
                for q_, x_ho_ in zip(batch_mixture, batch_x_ho)
            ]).mean()

            return 'neg MLL [mixt] {:8.04f}'.format(mean_neg_ave_log_marginal_likelihood.item())

        def edge_belief_mixture(params):
            '''
            L1 edge belief
            '''
           
            batch_particles = params['batch_x']
            batch_size = batch_particles.shape[0]
            batch_x_ho = [jnp.array(variants_target[b % n_variants].x_ho) for b in range(batch_size)]
            batch_g = [jnp.array(variants_target[b % n_variants].g) for b in range(batch_size)]

            batch_hard_g = self.particle_to_hard_g(batch_particles)
            batch_mixture = [
                particle_empirical_mixture(
                    hard_g, lambda g: self.eltwise_log_prob(g, b))
                for b, hard_g in zip(jnp.arange(batch_size), batch_hard_g)
            ]
            
            mean_edge_belief = jnp.array([
                l1_edge_belief(dist=q_, g=g_)
                for q_, g_ in zip(batch_mixture, batch_g)
            ]).mean()

            return 'edge belief [mixt] {:8.04f}'.format(mean_edge_belief.item())


        return {
            'neg_ave_log_marginal_likelihood':  neg_ave_log_marginal_likelihood_mixture,
            'edge_belief':  edge_belief_mixture,
        }


    def make_metrics_ground_truth(self, variants_log_posterior, mmd_kernel, n_mmd_samples=100, n_kl_samples=10, max_entries_per_batch=1e7):

        n_variants = len(variants_log_posterior)

        '''Initialization / JIT'''
        # MMD
        mmd = MaximumMeanDiscrepancy(kernel=mmd_kernel)

        '''Metrics'''
        # metrics
        def kl_metric_hard(params):
            '''
            Computes forward KL for particles during SVGD
            if finite = True, log q(x) for q(x) = 0 is treated as 1e-1000 instead of -inf to see progress
            '''
            batch_x = params['batch_x']
            batch_size = batch_x.shape[0]
            batch_hard_g = self.particle_to_hard_g(batch_x)
            mean_kl_q_p = jnp.array([
                kullback_leibler_dist(particle_empirical(hard_g), variants_log_posterior[b % n_variants], finite=True)
                for b, hard_g in zip(jnp.arange(batch_size), batch_hard_g)
            ]).mean()
            return 'KL(q,p): {:8.04f}'.format(mean_kl_q_p.item())

        def min_kl_metric_hard(params):
            '''
            Computes forward KL for particles during SVGD
            if finite = True, log q(x) for q(x) = 0 is treated as 1e-1000 instead of -inf to see progress
            '''
            batch_x = params['batch_x']
            batch_size = batch_x.shape[0]
            batch_hard_g = self.particle_to_hard_g(batch_x)
            min_kl_q_p = jnp.array([
                kullback_leibler_dist(particle_empirical(hard_g), variants_log_posterior[b % n_variants], finite=True)
                for b, hard_g in zip(jnp.arange(batch_size), batch_hard_g)
            ]).min()
            return 'min KL(q,p): {:8.04f}'.format(min_kl_q_p.item())

        def kl_metric_hard_mixture(params):
            '''
            Computes backward KL for particles during SVGD
            if finte = True, log q(x) for q(x) = 0 is treated as 1e-1000 instead of -inf to see progress
            '''
           
            batch_x = params['batch_x']
            batch_size = batch_x.shape[0]
            batch_hard_g = self.particle_to_hard_g(batch_x)
            mean_kl_q_p = jnp.array([
                kullback_leibler_dist(
                    particle_empirical_mixture(hard_g, lambda g: self.eltwise_log_prob(g, b)), 
                    variants_log_posterior[b % n_variants], 
                    finite=True)
                for b, hard_g in zip(jnp.arange(batch_size), batch_hard_g)
            ]).mean()
            return 'KL(q,p) [mixt]: {:8.04f}'.format(mean_kl_q_p)


        def mmd_metric_hard(params):
            '''
            Computes MMD(p, q) for particles during SVGD
            using Categorical ground truth samples
            '''
            batch_x = params['batch_x']
            batch_size = batch_x.shape[0]
            batch_hard_g = self.particle_to_hard_g(batch_x)
            batch_keys = params['batch_keys']

            squared_mmds = []
            for b, hard_g in enumerate(batch_hard_g):
                key, subk = random.split(batch_keys[b])
                mmd_g_samples = id2bit(random.categorical(subk, 
                    variants_log_posterior[b % n_variants][:, 1], 
                    shape=(n_mmd_samples,)), 
                self.n_vars)

                # MMD
                squared_mmds.append(mmd.squared_mmd(p_samples=mmd_g_samples, q_samples=hard_g))

            return 'MMD(p,q) [hard]: {:8.04f}'.format(jnp.mean(jnp.array(squared_mmds)))

        def cyclic_graph_count_hard(params):
            '''
            Computes number of graphs that are cyclic amongst hard graphs implied by x
            '''
            batch_x = params['batch_x']
            batch_size = batch_x.shape[0]
            batch_hard_g = self.particle_to_hard_g(batch_x)
            mean_dag_count = jnp.array([
                (eltwise_acyclic_constr(hard_g, self.n_vars)[0] == 0).sum()
                for hard_g in batch_hard_g
            ]).mean()
            return 'Cyclic: {:4.1f}'.format(n_particles - mean_dag_count)

        def unique_graph_count_hard(params):
            '''
            Computes number of unique graphs implied by x
            '''
            batch_x = params['batch_x']
            batch_hard_g = self.particle_to_hard_g(batch_x)
            mean_n_unique = jnp.array([
                len(jnp.unique(bit2id(hard_g)))
                for hard_g in batch_hard_g
            ]).mean()
            return 'Unique: {:4.1f}'.format(mean_n_unique)

        def alpha_print(params):
            return 'alpha: {:4.2f}'.format(params['alpha'])

        def beta_print(params):
            return 'beta: {:4.2f}'.format(params['beta'])

        def particle_norm_print(params):
            batch_size = params['batch_x'].shape[0]
            return 'norm(x): {:9.4f}'.format(jnp.linalg.norm(params['batch_x'].reshape(batch_size, -1), axis=1).mean(0))

        def phi_norm_print(params):
            batch_size = params['batch_x'].shape[0]
            return 'norm(phi): {:9.6f}'.format(jnp.linalg.norm(params['batch_phi'].reshape(batch_size, -1), axis=1).mean(0))

        return {
            'kl_metric_hard':           kl_metric_hard,
            'min_kl_metric_hard':       min_kl_metric_hard,
            'kl_metric_hard_mixture':   kl_metric_hard_mixture,
            'mmd_metric_hard':          mmd_metric_hard,
            'cyclic_graph_count_hard':  cyclic_graph_count_hard,
            'unique_graph_count_hard':  unique_graph_count_hard,
            'alpha_print':              alpha_print,
            'beta_print':               beta_print,
            'particle_norm_print':      particle_norm_print,
            'phi_norm_print':           phi_norm_print,
        }

