import jax.numpy as jnp
from jax import vmap, random, grad
from jax.scipy.special import logsumexp
from jax.ops import index, index_mul
from jax.nn import sigmoid, log_sigmoid
import jax.lax as lax
from jax.tree_util import tree_map, tree_multimap

from dibs.utils.graph import acyclic_constr_nograd
from dibs.utils.func import expand_by


class DiBS:
    """
    This class implements the backbone for DiBS: Differentiable Bayesian Structure Learning (Lorch et al., 2021)

    The code is implemented with `JAX` to allow just-in-time compilation of numpy-style code with `jit()` 
    and vectorization using `vmap()`. To allow for this, the tensor shapes throughout the algorithm have 
    to remain *unchanged* depending on the input, and thus the resulting code style may be unfamiliar 
    when not being used to this requirement.

    Args:
        target_log_prior: log p(G); differentiable log prior probability using the probabilities of edges (as implied by Z)
        target_log_joint_prob: log p(theta, D | G); differentiable or non-differentiable discrete log probability of target distribution
        alpha_linear (float): inverse temperature parameter schedule of sigmoid
        beta_linear (float): inverse temperature parameter schedule of prior
        n_grad_mc_samples (int): MC samples in gradient estimator for likelihood term p(theta, D | G)
        n_acyclicity_mc_samples (int): MC samples in gradient estimator for acyclicity constraint
        grad_estimator_z (str): gradient estimator d/dZ of expectation; choices: `score` or `reparam`
        score_function_baseline (float): weight of addition in score function baseline; == 0.0 corresponds to not using a baseline
        latent_prior_std (float): standard deviation of Gaussian prior over Z; defaults to 1/sqrt(k)
    """

    def __init__(self, *, target_log_prior, target_log_joint_prob, alpha_linear, beta_linear=1.0, tau=1.0,
                 n_grad_mc_samples=128, n_acyclicity_mc_samples=32, 
                 grad_estimator_z='reparam', score_function_baseline=0.0,
                 latent_prior_std=None, verbose=False):
        super(DiBS, self).__init__()

        self.target_log_prior = target_log_prior
        self.target_log_joint_prob = target_log_joint_prob
        self.alpha = lambda t: (alpha_linear * t)
        self.beta = lambda t: (beta_linear * t)
        self.tau = tau
        self.n_grad_mc_samples = n_grad_mc_samples
        self.n_acyclicity_mc_samples = n_acyclicity_mc_samples
        self.grad_estimator_z = grad_estimator_z
        self.score_function_baseline = score_function_baseline
        self.latent_prior_std = latent_prior_std
        self.verbose = verbose

    """
    Backbone functionality
    """

    def vec_to_mat(self, z, n_vars):
        """
        Reshapes particle to latent adjacency matrix form
            last dim gets shaped into matrix
        
        Args:
            w: flattened matrix of shape [..., d * d]

        Returns:
            matrix of shape [..., d, d]
        """
        return z.reshape(*z.shape[:-1], n_vars, n_vars)


    def mat_to_vec(self, w):
        """
        Reshapes latent adjacency matrix form to particle
            last two dims get flattened into vector
        
        Args:
            w: matrix of shape [..., d, d]
        
        Returns:
            flattened matrix of shape [..., d * d]
        """
        n_vars = w.shape[-1]
        return w.reshape(*w.shape[:-2], n_vars * n_vars)


    def particle_to_g_lim(self, z):
        """
        Returns g corresponding to alpha = infinity for particles `z`

        Args:
            z: latent variables [..., d, k, 2]

        Returns:
            graph adjacency matrices of shape [..., d, d]
        """
        u, v = z[..., 0], z[..., 1]
        scores = jnp.einsum('...ik,...jk->...ij', u, v)
        g_samples = (scores > 0).astype(jnp.int32)

        # zero diagonal
        g_samples = index_mul(g_samples, index[..., jnp.arange(scores.shape[-1]), jnp.arange(scores.shape[-1])], 0)
        return g_samples


    def sample_g(self, p, subk, n_samples):
        """
        Sample Bernoulli matrix according to matrix of probabilities

        Args:
            p: matrix of probabilities [d, d]
            n_samples: number of samples
            subk: rng key
        
        Returns:
            an array of matrices sampled according to `p` of shape [n_samples, d, d]
        """
        n_vars = p.shape[-1]
        g_samples = self.vec_to_mat(random.bernoulli(
            subk, p=self.mat_to_vec(p), shape=(n_samples, n_vars * n_vars)), n_vars).astype(jnp.int32)

        # mask diagonal since it is explicitly not modeled
        g_samples = index_mul(g_samples, index[..., jnp.arange(p.shape[-1]), jnp.arange(p.shape[-1])], 0)

        return g_samples

    def particle_to_soft_graph(self, z, eps, t):
        """ 
        Gumbel-softmax / concrete distribution using Logistic(0,1) samples `eps`

        Args:
            z: a single latent tensor Z of shape [d, k, 2]
            eps: random iid Logistic(0,1) noise  of shape [d, d] 
            t: step
        
        Returns:
            Gumbel-softmax sample of adjacency matrix [d, d]
        """
        scores = jnp.einsum('...ik,...jk->...ij', z[..., 0], z[..., 1])

        # soft reparameterization using gumbel-softmax/concrete distribution
        # eps ~ Logistic(0,1)
        soft_graph = sigmoid(self.tau * (eps + self.alpha(t) * scores))

        # mask diagonal since it is explicitly not modeled
        n_vars = soft_graph.shape[-1]
        soft_graph = index_mul(soft_graph, index[..., jnp.arange(n_vars), jnp.arange(n_vars)], 0.0)
        return soft_graph


    def particle_to_hard_graph(self, z, eps, t):
        """ 
        Bernoulli sample of G using probabilities implied by z

        Args:
            z: a single latent tensor [d, k, 2]
            eps: random iid Logistic(0,1) noise  of shape [d, d] 
            t: step
        
        Returns:
            Gumbel-max (hard) sample of adjacency matrix [d, d]
        """
        scores = jnp.einsum('...ik,...jk->...ij', z[..., 0], z[..., 1])

        # simply take hard limit of sigmoid in gumbel-softmax/concrete distribution
        hard_graph = ((self.tau * (eps + self.alpha(t) * scores)) > 0.0).astype(jnp.float64)

        # mask diagonal since it is explicitly not modeled
        n_vars = hard_graph.shape[-1]
        hard_graph = index_mul(hard_graph, index[..., jnp.arange(n_vars), jnp.arange(n_vars)], 0.0)
        return hard_graph


    """
    Generative graph model p(G | Z)
    """

    def edge_probs(self, z, t):
        """
        Edge probabilities encoded by latent representation 

        Args:
            z: latent tensors Z [..., d, k, 2]
            t: step
        
        Returns:
            edge probabilities of shape [..., d, d]
        """
        u, v = z[..., 0], z[..., 1]
        scores = jnp.einsum('...ik,...jk->...ij', u, v)
        probs =  sigmoid(self.alpha(t) * scores)

        # mask diagonal since it is explicitly not modeled
        probs = index_mul(probs, index[..., jnp.arange(probs.shape[-1]), jnp.arange(probs.shape[-1])], 0.0)
        return probs

    
    def edge_log_probs(self, z, t):
        """
        Edge log probabilities encoded by latent representation

        Args:
            z: latent tensors Z [..., d, k, 2]
            t: step

        Returns:
            tuple of tensors [..., d, d], [..., d, d] corresponding to log(p) and log(1-p)
        """
        u, v = z[..., 0], z[..., 1]
        scores = jnp.einsum('...ik,...jk->...ij', u, v)
        log_probs, log_probs_neg =  log_sigmoid(self.alpha(t) * scores), log_sigmoid(self.alpha(t) * -scores)

        # mask diagonal since it is explicitly not modeled
        # NOTE: this is not technically log(p), but the way `edge_log_probs_` is used, this is correct
        log_probs = index_mul(log_probs, index[..., jnp.arange(log_probs.shape[-1]), jnp.arange(log_probs.shape[-1])], 0.0)
        log_probs_neg = index_mul(log_probs_neg, index[..., jnp.arange(log_probs_neg.shape[-1]), jnp.arange(log_probs_neg.shape[-1])], 0.0)
        return log_probs, log_probs_neg



    def latent_log_prob(self, single_g, single_z, t):
        """
        Log likelihood of generative graph model

        Args:
            single_g: single graph adjacency matrix [d, d]    
            single_z: single latent tensor [d, k, 2]
            t: step
        
        Returns:
            log likelihood log p(G | Z) of shape [1,]
        """
        # [d, d], [d, d]
        log_p, log_1_p = self.edge_log_probs(single_z, t)

        # [d, d]
        log_prob_g_ij = single_g * log_p + (1 - single_g) * log_1_p

        # [1,] # diagonal is masked inside `edge_log_probs`
        log_prob_g = jnp.sum(log_prob_g_ij)

        return log_prob_g


    def eltwise_grad_latent_log_prob(self, gs, single_z, t):
        """
        Gradient of log likelihood of generative graph model w.r.t. Z
        i.e. d/dz log p(G | Z) 
        Batched over samples of G given a single Z.

        Args:
            gs: batch of graph matrices [n_graphs, d, d]
            single_z: latent variable [d, k, 2] 
            t: step

        Returns:
            batch of gradients of shape [n_graphs, d, k, 2]
        """
        dz_latent_log_prob = grad(self.latent_log_prob, 1)
        return vmap(dz_latent_log_prob, (0, None, None), 0)(gs, single_z, t)



    """
    Estimators for scores of log p(theta, D | Z) 
    """

    def eltwise_log_joint_prob(self, gs, single_theta, rng):
        """
        log p(data | G, theta) batched over samples of G

        Args:
            gs: batch of graphs [n_graphs, d, d]
            single_theta: single parameter PyTree
            rng:  [1, ]

        Returns:
            batch of logprobs [n_graphs, ]
        """

        return vmap(self.target_log_joint_prob, (0, None, None), 0)(gs, single_theta, rng)

    

    def log_joint_prob_soft(self, single_z, single_theta, eps, t, subk):
        """
        This is the composition of 
            log p(theta, D | G) 
        and
            G(Z, U)  (Gumbel-softmax graph sample given Z)

        Args:
            single_z: single latent tensor [d, k, 2]
            single_theta: single parameter PyTree
            eps: i.i.d Logistic noise of shpae [d, d] 
            t: step 
            subk: rng key

        Returns:
            logprob of shape [1, ]

        """
        soft_g_sample = self.particle_to_soft_graph(single_z, eps, t)
        return self.target_log_joint_prob(soft_g_sample, single_theta, subk)
    

    #
    # Estimators for score d/dZ log p(theta, D | Z)   
    # (i.e. w.r.t the latent embeddings Z for graph G)
    #

    def eltwise_grad_z_likelihood(self,  zs, thetas, baselines, t, subkeys):
        """
        Computes batch of estimators for score
            
            d/dZ log p(theta, D | Z) 

        Selects corresponding estimator used for the term `d/dZ E_p(G|Z)[ p(theta, D | G) ]`
        and executes it in batch.

        Args:
            zs: batch of latent tensors Z [n_particles, d, k, 2]
            thetas: batch of parameters PyTree with `n_particles` as leading dim
            baselines: array of score function baseline values of shape [n_particles, ]

        Returns:
            tuple: batch of (gradient estimates, baselines) of shapes [n_particles, d, k, 2], [n_particles, ]        
        """

        # select the chosen gradient estimator
        if self.grad_estimator_z == 'score':
            grad_z_likelihood = self.grad_z_likelihood_score_function

        elif self.grad_estimator_z == 'reparam':
            grad_z_likelihood = self.grad_z_likelihood_gumbel

        else:
            raise ValueError(f'Unknown gradient estimator `{self.grad_estimator_z}`')

        # vmap
        return vmap(grad_z_likelihood, (0, 0, 0, None, 0), (0, 0))(zs, thetas, baselines, t, subkeys)



    def grad_z_likelihood_score_function(self, single_z, single_theta, single_sf_baseline, t, subk):
        """
        Score function estimator (aka REINFORCE) for the score

            d/dZ log p(theta, D | Z) 

        This does not use d/dG log p(theta, D | G) and is hence applicable when not defined.
        Uses same G samples for expectations in numerator and denominator.

        Args:
            single_z: single latent tensor [d, k, 2]
            single_theta: single parameter PyTree
            single_sf_baseline: [1, ]
            t: step
            subk: rng key
        
        Returns:
            tuple gradient, baseline  [d, k, 2], [1, ]

        """

        # [d, d]
        p = self.edge_probs(single_z, t)
        n_vars, n_dim = single_z.shape[0:2]

        # [n_grad_mc_samples, d, d]
        subk, subk_ = random.split(subk)
        g_samples = self.sample_g(p, subk_, self.n_grad_mc_samples)

        # same MC samples for numerator and denominator
        n_mc_numerator = self.n_grad_mc_samples
        n_mc_denominator = self.n_grad_mc_samples

        # [n_grad_mc_samples, ] 
        subk, subk_ = random.split(subk)
        logprobs_numerator = self.eltwise_log_joint_prob(g_samples, single_theta, subk_)
        logprobs_denominator = logprobs_numerator

        # variance_reduction
        logprobs_numerator_adjusted = lax.cond(
            self.score_function_baseline <= 0.0,
            lambda _: logprobs_numerator,
            lambda _: logprobs_numerator - single_sf_baseline,
            operand=None)

        # [d * k * 2, n_grad_mc_samples]
        grad_z = self.eltwise_grad_latent_log_prob(g_samples, single_z, t)\
            .reshape(self.n_grad_mc_samples, n_vars * n_dim * 2)\
            .transpose((1, 0))

        # stable computation of exp/log/divide
        # [d * k * 2, ]  [d * k * 2, ]
        log_numerator, sign = logsumexp(a=logprobs_numerator_adjusted, b=grad_z, axis=1, return_sign=True)

        # []
        log_denominator = logsumexp(logprobs_denominator, axis=0)

        # [d * k * 2, ]
        stable_sf_grad = sign * jnp.exp(log_numerator - jnp.log(n_mc_numerator) - log_denominator + jnp.log(n_mc_denominator))

        # [d, k, 2]
        stable_sf_grad_shaped = stable_sf_grad.reshape(n_vars, n_dim, 2)

        # update baseline
        single_sf_baseline = (self.score_function_baseline * logprobs_numerator.mean(0) +
                            (1 - self.score_function_baseline) * single_sf_baseline)

        return stable_sf_grad_shaped, single_sf_baseline
        


    def grad_z_likelihood_gumbel(self, single_z, single_theta, single_sf_baseline, t, subk):
        """
        Reparameterization estimator for the score

            d/dZ log p(theta, D | Z) 
            
        Using the Gumbel-softmax / concrete distribution reparameterization trick.
        Uses same G samples for expectations in numerator and denominator.

        Args:
            single_z: single latent tensor [d, k, 2]
            single_theta: single parameter PyTree
            single_sf_baseline: [1, ]

        Returns:
            tuple: gradient, baseline of shape [d, k, 2], [1, ]

        """   
        n_vars = single_z.shape[0]

        # same MC samples for numerator and denominator
        n_mc_numerator = self.n_grad_mc_samples
        n_mc_denominator = self.n_grad_mc_samples

        # sample Logistic(0,1) as randomness in reparameterization
        subk, subk_ = random.split(subk)
        eps = random.logistic(subk_, shape=(self.n_grad_mc_samples, n_vars, n_vars))                

        # [n_grad_mc_samples, ]
        # since we don't backprop per se, it leaves us with the option of having
        # `soft` and `hard` versions for evaluating the non-grad p(.))
        subk, subk_ = random.split(subk)
       
        # [d, k, 2], [d, d], [n_grad_mc_samples, d, d], [1,], [1,] -> [n_grad_mc_samples]
        logprobs_numerator = vmap(self.log_joint_prob_soft, (None, None, 0, None, None), 0)(single_z, single_theta, eps, t, subk_) 
        logprobs_denominator = logprobs_numerator

        # [n_grad_mc_samples, d, k, 2]
        # d/dx log p(theta, D | G(x, eps)) for a batch of `eps` samples
        # use the same minibatch of data as for other log prob evaluation (if using minibatching)
        
        # [d, k, 2], [d, d], [n_grad_mc_samples, d, d], [1,], [1,] -> [n_grad_mc_samples, d, k, 2]
        grad_z = vmap(grad(self.log_joint_prob_soft, 0), (None, None, 0, None, None), 0)(single_z, single_theta, eps, t, subk_)

        # stable computation of exp/log/divide
        # [d, k, 2], [d, k, 2]
        log_numerator, sign = logsumexp(a=logprobs_numerator[:, None, None, None], b=grad_z, axis=0, return_sign=True)

        # []
        log_denominator = logsumexp(logprobs_denominator, axis=0)

        # [d, k, 2]
        stable_grad = sign * jnp.exp(log_numerator - jnp.log(n_mc_numerator) - log_denominator + jnp.log(n_mc_denominator))

        return stable_grad, single_sf_baseline


    #
    # Estimators for score d/dtheta log p(theta, D | Z) 
    # (i.e. w.r.t the conditional distribution parameters)
    #

    def eltwise_grad_theta_likelihood(self, zs, thetas, t, subk):
        """
        Computes batch of estimators for the score
            
            d/dtheta log p(theta, D | Z) 

        (i.e. w.r.t the conditional distribution parameters)

        This does not use d/dG log p(theta, D | G) and is hence applicable when not defined.
        Analogous to `eltwise_grad_z_likelihood` but w.r.t theta

        Args:
            zs: batch of latent tensors Z [n_particles, d, k, 2]
            thetas: batch of parameter PyTree with `n_mc_samples` as leading dim

        Returns:
            batch of gradients in form of PyTree with `n_particles` as leading dim     

        """
        return vmap(self.grad_theta_likelihood, (0, 0, None, None), 0)(zs, thetas, t, subk)


    def grad_theta_likelihood(self, single_z, single_theta, t, subk):
        """
        Computes Monte Carlo estimator for the score 
            
            d/dtheta log p(theta, D | Z) 

        Uses hard samples of G; reparameterization like for d/dZ is also possible
        Uses same G samples for expectations in numerator and denominator.

        Args:
            single_z: single latent tensor [d, k, 2]
            single_theta: single parameter PyTree
            t: step
            subk: rng key

        Returns:
            parameter gradient PyTree

        """

        # [d, d]
        p = self.edge_probs(single_z, t)

        # [n_grad_mc_samples, d, d]
        g_samples = self.sample_g(p, subk, self.n_grad_mc_samples)

        # same MC samples for numerator and denominator
        n_mc_numerator = self.n_grad_mc_samples
        n_mc_denominator = self.n_grad_mc_samples

        # [n_mc_numerator, ] 
        subk, subk_ = random.split(subk)
        logprobs_numerator = self.eltwise_log_joint_prob(g_samples, single_theta, subk_)
        logprobs_denominator = logprobs_numerator

        # PyTree  shape of `single_theta` with additional leading dimension [n_mc_numerator, ...]
        # d/dtheta log p(theta, D | G) for a batch of G samples
        # use the same minibatch of data as for other log prob evaluation (if using minibatching)
        grad_theta_log_joint_prob = grad(self.target_log_joint_prob, 1)
        grad_theta = vmap(grad_theta_log_joint_prob, (0, None, None), 0)(g_samples, single_theta, subk_)

        # stable computation of exp/log/divide and PyTree compatible
        # sums over MC graph samples dimension to get MC gradient estimate of theta
        # original PyTree shape of `single_theta`
        log_numerator = tree_map(
            lambda leaf_theta: 
                logsumexp(a=expand_by(logprobs_numerator, leaf_theta.ndim - 1), b=leaf_theta, axis=0, return_sign=True)[0], 
            grad_theta)

        # original PyTree shape of `single_theta`
        sign = tree_map(
            lambda leaf_theta:
                logsumexp(a=expand_by(logprobs_numerator, leaf_theta.ndim - 1), b=leaf_theta, axis=0, return_sign=True)[1], 
            grad_theta)

        # []
        log_denominator = logsumexp(logprobs_denominator, axis=0)

        # original PyTree shape of `single_theta`
        stable_grad = tree_multimap(
            lambda sign_leaf_theta, log_leaf_theta: 
                (sign_leaf_theta * jnp.exp(log_leaf_theta - jnp.log(n_mc_numerator) - log_denominator + jnp.log(n_mc_denominator))), 
            sign, log_numerator)

        return stable_grad


    """
    Estimators for score d/dZ log p(Z) 
    """
    def constraint_gumbel(self, single_z, single_eps, t):
        """ 
        Evaluates continuous acyclicity constraint using 
        Gumbel-softmax instead of Bernoulli samples

        Args:
            single_z: single latent tensor [d, k, 2]
            single_eps: i.i.d. Logistic noise of shape [d, d] for Gumbel-softmax
            t: step
        
        Returns:
            constraint value of shape [1,]
        """
        n_vars = single_z.shape[0]
        G = self.particle_to_soft_graph(single_z, single_eps, t)
        h = acyclic_constr_nograd(G, n_vars)
        return h


    def grad_constraint_gumbel(self, single_z, key, t):
        """
        Reparameterization estimator for the gradient

           d/dZ E_p(G|Z) [constraint(G)]
            
        Using the Gumbel-softmax / concrete distribution reparameterization trick.

        Args:
            z: single latent tensor [d, k, 2]                
            key: rng key [1,]    
            t: step

        Returns         
            gradient of constraint [d, k, 2] 
        """
        n_vars = single_z.shape[0]
        
        # [n_mc_samples, d, d]
        eps = random.logistic(key, shape=(self.n_acyclicity_mc_samples, n_vars, n_vars))

        # [n_mc_samples, d, k, 2]
        mc_gradient_samples = vmap(grad(self.constraint_gumbel, 0), (None, 0, None), 0)(single_z, eps, t)

        # [d, k, 2]
        return mc_gradient_samples.mean(0)


    def target_log_prior_particle(self, single_z, t):
        """
        log p(Z) approx. log p(G) via edge probabilities

        Args:
            single_z: single latent tensor [d, k, 2]
            t: step

        Returns:
            log prior graph probability [1,] log p(G) evaluated with G_\alpha(Z)
                i.e. with the edge probabilities   
        """
        # [d, d] # masking is done inside `edge_probs`
        single_soft_g = self.edge_probs(single_z, t)

        # [1, ]
        return self.target_log_prior(single_soft_g)


    def eltwise_grad_latent_prior(self, zs, subkeys, t):
        """
        Computes batch of estimators for the score

            d/dZ log p(Z)
        
        where log p(Z) = - beta(t) E_p(G|Z) [constraint(G)]
                         + log Gaussian(Z)
                         + log f(Z) 
        
        and f(Z) is an additional prior factor.

        Args:
            zs: single latent tensor  [n_particles, d, k, 2]
            subkeys: batch of rng keys [n_particles, ...]

        Returns:
            batch of gradients of shape [n_particles, d, k, 2]

        """

        # log f(Z) term
        # [d, k, 2], [1,] -> [d, k, 2]
        grad_target_log_prior_particle = grad(self.target_log_prior_particle, 0)

        # [n_particles, d, k, 2], [1,] -> [n_particles, d, k, 2]
        grad_prior_z = vmap(grad_target_log_prior_particle, (0, None), 0)(zs, t)

        # constraint term
        # [n_particles, d, k, 2], [n_particles,], [1,] -> [n_particles, d, k, 2]
        eltwise_grad_constraint = vmap(self.grad_constraint_gumbel, (0, 0, None), 0)(zs, subkeys, t)

        return - self.beta(t) * eltwise_grad_constraint \
               - zs / (self.latent_prior_std ** 2.0) \
               + grad_prior_z 
            
        
