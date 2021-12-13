import functools
import numpy as onp

import jax
import jax.numpy as jnp
from jax import jit, vmap, random, grad
from jax.tree_util import tree_map, tree_multimap
from jax.scipy.special import logsumexp
try:
    from jax.example_libraries import optimizers
except ImportError:
    # for jax <= 2.24
    from jax.experimental import optimizers


from dibs.inference.dibs import DiBS
from dibs.kernel import AdditiveFrobeniusSEKernel, JointAdditiveFrobeniusSEKernel
from dibs.metrics import ParticleDistribution
from dibs.utils.func import expand_by


class MarginalDiBS(DiBS):
    """
    This class implements DiBS: Differentiable Bayesian Structure Learning (Lorch et al., 2021)
    instantiated using Stein Variational Gradient Descent (SVGD) (Liu and Wang, 2016) as the underlying inference method.
    An SVGD update of vector v is defined as

        phi(v) = 1/n_particles sum_u k(v, u) d/du log p(u) + d/du k(u, v)

    This class implements //marginal// inference of the posterior p(G | D).
    For joint inference of p(G, theta | D), use the class `JointDiBS`

    Args:
        x: observations of shape [n_observations, n_vars]
        inference_model: Bayes net inference model defining prior and marginal likelihood underlying the inferred posterior

        kernel: class of kernel with differentiable evaluation function `eval`
        kernel_param: kwargs to instantiate `kernel`
        optimizer: optimizer identifier str
        optimizer_param: kwargs to instantiate `optimizer`

        alpha_linear (float): inverse temperature parameter schedule of sigmoid
        beta_linear (float): inverse temperature parameter schedule of acyclicity prior
        tau (float): Gumbel-softmax relaxation temperature

        n_grad_mc_samples (int): number of Monte Carlo samples in gradient estimator for likelihood term p(theta, D | G)
        n_acyclicity_mc_samples (int): number of Monte Carlo samples in gradient estimator of acyclicity prior
        grad_estimator_z (str): gradient estimator d/dZ of expectation; choices: `score` or `reparam`
        score_function_baseline (float): weight of addition in score function estimator baseline
        latent_prior_std (float): standard deviation of Gaussian prior over Z; defaults to 1/sqrt(k)

    """

    def __init__(self, *, x, inference_model,
                 kernel=AdditiveFrobeniusSEKernel, kernel_param=None,
                 optimizer="rmsprop", optimizer_param=None,
                 alpha_linear=1.0, beta_linear=1.0, tau=1.0,
                 n_grad_mc_samples=128, n_acyclicity_mc_samples=32,
                 grad_estimator_z="score", score_function_baseline=0.0,
                 latent_prior_std=None, verbose=False):

        # handle mutable default args
        if kernel_param is None:
            kernel_param = {"h": 5.0}
        if optimizer_param is None:
            optimizer_param = {"stepsize": 0.005}

        # init DiBS superclass methods
        super(MarginalDiBS, self).__init__(
            x=x,
            log_graph_prior=inference_model.log_graph_prior,
            log_joint_prob=inference_model.observational_log_marginal_prob,
            alpha_linear=alpha_linear,
            beta_linear=beta_linear,
            tau=tau,
            n_grad_mc_samples=n_grad_mc_samples,
            n_acyclicity_mc_samples=n_acyclicity_mc_samples,
            grad_estimator_z=grad_estimator_z,
            score_function_baseline=score_function_baseline,
            latent_prior_std=latent_prior_std,
            verbose=verbose,
        )

        self.inference_model = inference_model
        self.eltwise_log_marginal_likelihood = vmap(lambda g, x_ho:
                                                    inference_model.observational_log_marginal_prob(g, None, x_ho,
                                                                                                    None), (0, None), 0)

        self.kernel = kernel(**kernel_param)

        if optimizer == 'gd':
            self.opt = optimizers.sgd(optimizer_param['stepsize'])
        elif optimizer == 'rmsprop':
            self.opt = optimizers.rmsprop(optimizer_param['stepsize'])
        else:
            raise ValueError()

    def _sample_initial_random_particles(self, *, key, n_particles, n_dim=None):
        """
        Samples random particles to initialize SVGD

        Args:
            key: rng key
            n_particles: number of particles inferred
            n_dim: size of latent dimension `k`. Defaults to `n_vars`, s.t. k == d

        Returns:
            z: batch of latent tensors [n_particles, d, k, 2]
        """
        # default full rank
        if n_dim is None:
            n_dim = self.n_vars

        # like prior
        std = self.latent_prior_std or (1.0 / jnp.sqrt(n_dim))

        # sample
        key, subk = random.split(key)
        z = random.normal(subk, shape=(n_particles, self.n_vars, n_dim, 2)) * std

        return z

    def _f_kernel(self, x_latent, y_latent):
        """
        Evaluates kernel

        Args:
            x_latent: latent tensor [d, k, 2]
            y_latent: latent tensor [d, k, 2]

        Returns:
            [1, ] kernel value
        """
        return self.kernel.eval(x=x_latent, y=y_latent)

    def _f_kernel_mat(self, x_latents, y_latents):
        """
        Computes pairwise kernel matrix

        Args:
            x_latents: latent tensor [A, d, k, 2]
            y_latents: latent tensor [B, d, k, 2]

        Returns:
            [A, B] kernel values
        """
        return vmap(vmap(self._f_kernel, (None, 0), 0), (0, None), 0)(x_latents, y_latents)

    def _eltwise_grad_kernel_z(self, x_latents, y_latent):
        """
        Computes gradient d/dz k(z, z') elementwise for each provided particle z

        Args:
            x_latents: batch of latent particles [n_particles, d, k, 2]
            y_latent: single latent particle [d, k, 2] (z')

        Returns:
            batch of gradients for latent tensors Z [n_particles, d, k, 2]
        """
        grad_kernel_z = grad(self._f_kernel, 0)
        return vmap(grad_kernel_z, (0, None), 0)(x_latents, y_latent)

    def _z_update(self, single_z, kxx, z, grad_log_prob_z):
        """
        Computes SVGD update for `single_z` particlee given the kernel values
        `kxx` and the d/dz gradients of the target density for each of the available particles

        Args:
            single_z: single latent tensor Z [d, k, 2], which is the Z particle being updated
            kxx: pairwise kernel values for all particles [n_particles, n_particles]
            z:  all latent tensor Z particles [n_particles, d, k, 2]
            grad_log_prob_z: gradients of all Z particles w.r.t target density  [n_particles, d, k, 2]

        Returns
            transform vector of shape [d, k, 2] for the Z particle being updated

        """

        # compute terms in sum
        weighted_gradient_ascent = kxx[..., None, None, None] * grad_log_prob_z
        repulsion = self._eltwise_grad_kernel_z(z, single_z)

        # average and negate (for optimizer)
        return - (weighted_gradient_ascent + repulsion).mean(axis=0)

    def _parallel_update_z(self, *args):
        """
        Parallelizes `z_update` for all available particles
        Otherwise, same inputs as `z_update`.
        """
        return vmap(self._z_update, (0, 1, None, None), 0)(*args)

    def _svgd_step(self, t, opt_state_z, key, sf_baseline):
        """
        Performs a single SVGD step in the DiBS framework, updating all Z particles jointly.

        Args:
            t: step
            opt_state_z: optimizer state for latent Z particles; contains [n_particles, d, k, 2]
            key: prng key
            sf_baseline: batch of baseline values in case score function gradient is used [n_particles, ]

        Returns:
            the updated inputs
        """

        z = self.get_params(opt_state_z)  # [n_particles, d, k, 2]
        n_particles = z.shape[0]

        # d/dz log p(D | z)
        key, *batch_subk = random.split(key, n_particles + 1)
        dz_log_likelihood, sf_baseline = self.eltwise_grad_z_likelihood(z, None, sf_baseline, t, jnp.array(batch_subk))
        # here `None` is a placeholder for theta (in the joint inference case)
        # since this is an inherited function from the general `DiBS` class

        # d/dz log p(z) (acyclicity)
        key, *batch_subk = random.split(key, n_particles + 1)
        dz_log_prior = self.eltwise_grad_latent_prior(z, jnp.array(batch_subk), t)

        # d/dz log p(z, D) = d/dz log p(z)  + log p(D | z)
        dz_log_prob = dz_log_prior + dz_log_likelihood

        # k(z, z) for all particles
        kxx = self._f_kernel_mat(z, z)

        # transformation phi() applied in batch to each particle individually
        phi_z = self._parallel_update_z(z, kxx, z, dz_log_prob)

        # apply transformation
        # `x += stepsize * phi`; the phi returned is negated for SVGD
        opt_state_z = self.opt_update(t, phi_z, opt_state_z)

        return opt_state_z, key, sf_baseline

    # this is the crucial @jit
    @functools.partial(jit, static_argnums=(0, 2))
    def _svgd_loop(self, start, n_steps, init):
        return jax.lax.fori_loop(start, start + n_steps, lambda i, args: self._svgd_step(i, *args), init)

    def sample(self, *, key, n_particles, steps, n_dim_particles=None, callback=None, callback_every=None):
        """
        Use SVGD to sample `n_particles` particles G from the marginal posterior p(G | D) as
        defined by the BN model `self.inference_model`

        Arguments:
            key: prng key
            n_particles (int): number of particles to sample
            steps (int): number of SVGD steps performed
            n_dim_particles (int): latent dimensionality k of particles Z; default is `n_vars`
            callback: function to be called every `callback_every` steps of SVGD.
            callback_every: if `None`, `callback` is only called after particle updates have finished

        Returns:
            particles_g: [n_particles, n_vars, n_vars]

        """

        # randomly sample initial particles
        key, subk = random.split(key)
        init_z = self._sample_initial_random_particles(key=subk, n_particles=n_particles, n_dim=n_dim_particles)

        # initialize score function baseline (one for each particle)
        n_particles, _, n_dim, _ = init_z.shape
        sf_baseline = jnp.zeros(n_particles)

        if self.latent_prior_std is None:
            self.latent_prior_std = 1.0 / jnp.sqrt(n_dim)

        # maintain updated particles with optimizer state
        opt_init, self.opt_update, get_params = self.opt
        self.get_params = jit(get_params)
        opt_state_z = opt_init(init_z)

        """Execute particle update steps for all particles in parallel using `vmap` functions"""
        # faster if for-loop is functionally pure and compiled, so only interrupt for callback
        callback_every = callback_every or steps
        for t in range(0, steps, callback_every):

            # perform sequence of SVGD steps
            opt_state_z, key, sf_baseline = self._svgd_loop(t, callback_every, (opt_state_z, key, sf_baseline))

            # callback
            if callback:
                z = self.get_params(opt_state_z)
                callback(
                    dibs=self,
                    t=t + callback_every,
                    zs=z,
                )

        # retrieve transported particles
        z_final = jax.device_get(self.get_params(opt_state_z))

        # as alpha is large, we can convert the latents Z to their corresponding graphs G
        g_final = self.particle_to_g_lim(z_final)
        return g_final

    def get_empirical(self, g):
        """
        Converts batch of binary (adjacency) matrices into empirical particle distribution

        Args:
            g: [N, d, d] with {0,1} values

        Returns:
            ParticleDistribution
        """
        N, _, _ = g.shape
        unique, counts = onp.unique(g, axis=0, return_counts=True)

        # empirical distribution using counts
        logp = jnp.log(counts) - jnp.log(N)

        return ParticleDistribution(logp=logp, g=g)

    def get_mixture(self, g):
        """
        Converts batch of binary (adjacency) matrices into mixture particle distribution,
        where mixture weights correspond to unnormalized target (i.e. posterior) probabilities

        Args:
           g: [N, d, d] with {0,1} values

        Returns:
           ParticleDistribution

        """

        N, _, _ = g.shape

        # mixture weighted by respective marginal probabilities
        eltwise_log_marginal_target = vmap(lambda single_g: self.log_joint_prob(single_g, None, self.x, None), 0, 0)
        logp = eltwise_log_marginal_target(g)
        logp -= logsumexp(logp)

        return ParticleDistribution(logp=logp, g=g)




class JointDiBS(DiBS):
    """
    This class implements DiBS: Differentiable Bayesian Structure Learning (Lorch et al., 2021)
    instantiated using Stein Variational Gradient Descent (SVGD) (Liu and Wang, 2016) as the underlying inference method.
    An SVGD update of vector v is defined as

        phi(v) = 1/n_particles sum_u k(v, u) d/du log p(u) + d/du k(u, v)
        
    This class implements //joint// inference of the posterior p(G, theta | D).
    For marginal inference of p(G | D), use the class `MarginalDiBS`

    Args:
        x: observations of shape [n_observations, n_vars]
        inference_model: Bayes net inference model defining prior and likelihood underlying the inferred posterior

        kernel: class of kernel with differentiable evaluation function `eval`
        kernel_param: kwargs to instantiate `kernel`
        optimizer: optimizer identifier str
        optimizer_param: kwargs to instantiate `optimizer`

        alpha_linear (float): inverse temperature parameter schedule of sigmoid
        beta_linear (float): inverse temperature parameter schedule of acyclicity prior
        tau (float): Gumbel-softmax relaxation temperature

        n_grad_mc_samples (int): number of Monte Carlo samples in gradient estimator for likelihood term p(theta, D | G)
        n_acyclicity_mc_samples (int): number of Monte Carlo samples in gradient estimator of acyclicity prior
        grad_estimator_z (str): gradient estimator d/dZ of expectation; choices: `score` or `reparam`
        score_function_baseline (float): weight of addition in score function estimator baseline
        latent_prior_std (float): standard deviation of Gaussian prior over Z; defaults to 1/sqrt(k)

    """

    def __init__(self, *, x, inference_model,
                 kernel=JointAdditiveFrobeniusSEKernel, kernel_param=None,
                 optimizer="rmsprop", optimizer_param=None,
                 alpha_linear=0.05, beta_linear=1.0, tau=1.0,
                 n_grad_mc_samples=128, n_acyclicity_mc_samples=32,
                 grad_estimator_z="reparam", score_function_baseline=0.0,
                 latent_prior_std=None, verbose=False):

        # handle mutable default args
        if kernel_param is None:
            kernel_param = {"h_latent": 5.0, "h_theta": 500.0}
        if optimizer_param is None:
            optimizer_param = {"stepsize": 0.005}

        # init DiBS superclass methods
        super(JointDiBS, self).__init__(
            x=x,
            log_graph_prior=inference_model.log_graph_prior,
            log_joint_prob=inference_model.observational_log_joint_prob,
            alpha_linear=alpha_linear,
            beta_linear=beta_linear,
            tau=tau,
            n_grad_mc_samples=n_grad_mc_samples,
            n_acyclicity_mc_samples=n_acyclicity_mc_samples,
            grad_estimator_z=grad_estimator_z,
            score_function_baseline=score_function_baseline,
            latent_prior_std=latent_prior_std,
            verbose=verbose,
        )

        self.inference_model = inference_model
        self.eltwise_log_likelihood = vmap(lambda g, theta, x_ho:
            inference_model.observational_log_joint_prob(g, theta, x_ho, None), (0, 0, None), 0)

        self.kernel = kernel(**kernel_param)

        if optimizer == 'gd':
            self.opt = optimizers.sgd(optimizer_param['stepsize'])
        elif optimizer == 'rmsprop':
            self.opt = optimizers.rmsprop(optimizer_param['stepsize'])
        else:
            raise ValueError()
       

    def _sample_initial_random_particles(self, *, key, n_particles, n_dim=None):
        """
        Samples random particles to initialize SVGD

        Args:
            key: rng key
            n_particles: number of particles inferred
            n_dim: size of latent dimension `k`. Defaults to `n_vars`, s.t. k == d

        Returns:
            z: batch of latent tensors [n_particles, d, k, 2]
            theta: batch of parameters PyTree with leading dim `n_particles`
        
        """
        # default full rank
        if n_dim is None:
            n_dim = self.n_vars
        
        # std like Gaussian prior over Z           
        std = self.latent_prior_std  or (1.0 / jnp.sqrt(n_dim))

        # sample from parameter prior
        key, subk = random.split(key)
        z = random.normal(subk, shape=(n_particles, self.n_vars, n_dim, 2)) * std

        key, subk = random.split(key)
        theta = self.inference_model.sample_parameters(key=subk, n_particles=n_particles, n_vars=self.n_vars)

        return z, theta


    def _f_kernel(self, x_latent, x_theta, y_latent, y_theta):
        """
        Evaluates kernel

        Args:
            x_latent: latent tensor [d, k, 2]
            x_theta: parameter PyTree 
            y_latent: latent tensor [d, k, 2]
            y_theta: parameter PyTree 

        Returns:
            kernel value
        """
        return self.kernel.eval(
            x_latent=x_latent, x_theta=x_theta,
            y_latent=y_latent, y_theta=y_theta)
    

    def _f_kernel_mat(self, x_latents, x_thetas, y_latents, y_thetas):
        """
        Computes pairwise kernel matrix

        Args:
            x_latents: latent tensor [A, d, k, 2]
            x_thetas: parameter PyTree with batch size A as leading dim
            y_latents: latent tensor [B, d, k, 2]
            y_thetas: parameter PyTree with batch size B as leading dim

        Returns:
            [A, B] kernel values
        """
        return vmap(vmap(self._f_kernel, (None, None, 0, 0), 0),
                    (0, 0, None, None), 0)(x_latents, x_thetas, y_latents, y_thetas)

    
    def _eltwise_grad_kernel_z(self, x_latents, x_thetas, y_latent, y_theta):
        """
        Computes gradient d/dz k((z, theta), (z', theta')) elementwise for each provided particle (z, theta)

        Args:
            x_latents: batch of latent particles [n_particles, d, k, 2]
            x_thetas: batch of parameter PyTree with leading dim `n_particles`
            y_latent: single latent particle [d, k, 2] (z')
            y_theta: single parameter PyTree (theta')

        Returns:
            batch of gradients for latent tensors Z [n_particles, d, k, 2]
        
        """        
        grad_kernel_z = grad(self._f_kernel, 0)
        return vmap(grad_kernel_z, (0, 0, None, None), 0)(x_latents, x_thetas, y_latent, y_theta)


    def _eltwise_grad_kernel_theta(self, x_latents, x_thetas, y_latent, y_theta):
        """
        Computes gradient d/dtheta k((z, theta), (z', theta')) elementwise for each provided particle (z, theta)

        Args:
            x_latents: batch of latent particles [n_particles, d, k, 2]
            x_thetas: batch of parameter PyTree with leading dim `n_particles`
            y_latent: single latent particle [d, k, 2] (z')
            y_theta: single parameter PyTree (theta')

        Returns:
            batch of gradients for parameters (PyTree with leading dim `n_particles`)
        """
        grad_kernel_theta = grad(self._f_kernel, 1)
        return vmap(grad_kernel_theta, (0, 0, None, None), 0)(x_latents, x_thetas, y_latent, y_theta)


    def _z_update(self, single_z, single_theta, kxx, z, theta, grad_log_prob_z):
        """
        Computes SVGD update for `single_z` of a (single_z, single_theta) tuple given the kernel values 
        `kxx` and the d/dz gradients of the target density for each of the available particles 

        Args:
            single_z: single latent tensor Z [d, k, 2], which is the Z particle being updated
            single_theta: single parameter PyTree, the theta particle of the Z particle being updated
            kxx: pairwise kernel values for all particles [n_particles, n_particles]  
            z:  all latent tensor Z particles [n_particles, d, k, 2] 
            theta: all theta particles as PyTree with leading dim `n_particles` 
            grad_log_prob_z: gradients of all Z particles w.r.t target density  [n_particles, d, k, 2]  

        Returns
            transform vector of shape [d, k, 2] for the Z particle being updated        
        """
    
        # compute terms in sum
        weighted_gradient_ascent = kxx[..., None, None, None] * grad_log_prob_z
        repulsion = self._eltwise_grad_kernel_z(z, theta, single_z, single_theta)

        # average and negate (for optimizer)
        return - (weighted_gradient_ascent + repulsion).mean(axis=0)

    def _parallel_update_z(self, *args):
        """
        Parallelizes `z_update` for all available particles
        Otherwise, same inputs as `z_update`.
        """
        return vmap(self._z_update, (0, 0, 1, None, None, None), 0)(*args)


    def _theta_update(self, single_z, single_theta, kxx, z, theta, grad_log_prob_theta):
        """
        Computes SVGD update for `single_theta` of a (single_z, single_theta) tuple given the kernel values 
        `kxx` and the d/dtheta gradients of the target density for each of the available particles 

        Args:
            single_z: single latent tensor Z [d, k, 2], the Z particle of the theta particle being updated
            single_theta: single parameter PyTree being updated
            kxx: pairwise kernel values for all particles [n_particles, n_particles]  
            z:  all latent tensor Z particles [n_particles, d, k, 2] 
            theta: all theta particles as PyTree with leading dim `n_particles` 
            grad_log_prob_theta: gradients of all theta particles w.r.t target density 
                PyTree with leading dim `n_particles

        Returns:
            transform vector PyTree with leading dim `n_particles` for the theta particle being updated   
        """
    
        # compute terms in sum
        weighted_gradient_ascent = tree_map(
            lambda leaf_theta_grad: 
                expand_by(kxx, leaf_theta_grad.ndim - 1) * leaf_theta_grad, 
            grad_log_prob_theta)
        
        repulsion = self._eltwise_grad_kernel_theta(z, theta, single_z, single_theta)

        # average and negate (for optimizer)
        return  tree_multimap(
            lambda grad_asc_leaf, repuls_leaf: 
                - (grad_asc_leaf + repuls_leaf).mean(axis=0), 
            weighted_gradient_ascent, 
            repulsion)


    def _parallel_update_theta(self, *args):
        """
        Parallelizes `theta_update` for all available particles
        Otherwise, same inputs as `theta_update`.
        """
        return vmap(self._theta_update, (0, 0, 1, None, None, None), 0)(*args)


    def _svgd_step(self, t, opt_state_z, opt_state_theta, key, sf_baseline):
        """
        Performs a single SVGD step in the DiBS framework, updating Z and theta jointly.
        
        Args:
            t: step
            opt_state_z: optimizer state for latent Z particles; contains [n_particles, d, k, 2]
            opt_state_theta: optimizer state for theta particles; contains PyTree with `n_particles` leading dim
            key: prng key
            sf_baseline: batch of baseline values in case score function gradient is used [n_particles, ]

        Returns:
            the updated inputs
        """

        z = self.get_params(opt_state_z) # [n_particles, d, k, 2]
        theta = self.get_params(opt_state_theta) # PyTree with `n_particles` leading dim
        n_particles = z.shape[0]

        # d/dtheta log p(theta, D | z)
        key, subk = random.split(key)
        dtheta_log_prob = self.eltwise_grad_theta_likelihood(z, theta, t, subk)

        # d/dz log p(theta, D | z)
        key, *batch_subk = random.split(key, n_particles + 1)
        dz_log_likelihood, sf_baseline = self.eltwise_grad_z_likelihood(z, theta, sf_baseline, t, jnp.array(batch_subk))

        # d/dz log p(z) (acyclicity)
        key, *batch_subk = random.split(key, n_particles + 1)
        dz_log_prior = self.eltwise_grad_latent_prior(z, jnp.array(batch_subk), t)

        # d/dz log p(z, theta, D) = d/dz log p(z)  + log p(theta, D | z) 
        dz_log_prob = dz_log_prior + dz_log_likelihood
        
        # k((z, theta), (z, theta)) for all particles
        kxx = self._f_kernel_mat(z, theta, z, theta)

        # transformation phi() applied in batch to each particle individually
        phi_z = self._parallel_update_z(z, theta, kxx, z, theta, dz_log_prob)
        phi_theta = self._parallel_update_theta(z, theta, kxx, z, theta, dtheta_log_prob)

        # apply transformation
        # `x += stepsize * phi`; the phi returned is negated for SVGD
        opt_state_z = self.opt_update(t, phi_z, opt_state_z)
        opt_state_theta = self.opt_update(t, phi_theta, opt_state_theta)

        return opt_state_z, opt_state_theta, key, sf_baseline
    
    
    # this is the crucial @jit
    @functools.partial(jit, static_argnums=(0, 2))
    def _svgd_loop(self, start, n_steps, init):
        return jax.lax.fori_loop(start, start + n_steps, lambda i, args: self._svgd_step(i, *args), init)


    def sample(self, *, key, n_particles, steps, n_dim_particles=None, callback=None, callback_every=None):
        """
        Use SVGD to sample `n_particles` particles (G, theta) from the joint posterior p(G, theta | D) as
        defined by the BN model `self.inference_model`

        Arguments:
            key: prng key
            n_particles (int): number of particles to sample
            steps (int): number of SVGD steps performed
            n_dim_particles (int): latent dimensionality k of particles Z; default is `n_vars`
            callback: function to be called every `callback_every` steps of SVGD.
            callback_every: if `None`, `callback` is only called after particle updates have finished

        Returns:
            particles_g: [n_particles, n_vars, n_vars]
            particles_theta: PyTree of parameters with leading dimension `n_particles`
           
        """

        # randomly sample initial particles
        key, subk = random.split(key)
        init_z, init_theta = self._sample_initial_random_particles(key=subk, n_particles=n_particles,
                                                                   n_dim=n_dim_particles)

        # initialize score function baseline (one for each particle)
        n_particles, _, n_dim, _ = init_z.shape
        sf_baseline = jnp.zeros(n_particles)

        if self.latent_prior_std is None:
            self.latent_prior_std = 1.0 / jnp.sqrt(n_dim)

        # maintain updated particles with optimizer state
        opt_init, self.opt_update, get_params = self.opt
        self.get_params = jit(get_params)
        opt_state_z = opt_init(init_z)
        opt_state_theta = opt_init(init_theta)

        """Execute particle update steps for all particles in parallel using `vmap` functions"""
        # faster if for-loop is functionally pure and compiled, so only interrupt for callback
        callback_every = callback_every or steps
        for t in range(0, steps, callback_every):

            # perform sequence of SVGD steps
            opt_state_z, opt_state_theta, key, sf_baseline = self._svgd_loop(t, callback_every,
                                                                             (opt_state_z, opt_state_theta, key, sf_baseline))

            # callback
            if callback:
                z = self.get_params(opt_state_z)
                theta = self.get_params(opt_state_theta)
                callback(
                    dibs=self,
                    t=t + callback_every,
                    zs=z,
                    thetas=theta,
                )

        # retrieve transported particles
        z_final = jax.device_get(self.get_params(opt_state_z))
        theta_final = jax.device_get(self.get_params(opt_state_theta))

        # as alpha is large, we can convert the latents Z to their corresponding graphs G
        g_final = self.particle_to_g_lim(z_final)
        return g_final, theta_final


    def get_empirical(self, g, theta):
        """
        Converts batch of binary (adjacency) matrices and parameters into empirical particle distribution

        Args:
            g: [N, d, d] with {0,1} values
            theta: PyTree with leading dim `N`

        Returns:
            ParticleDistribution
        """
        N, _, _ = g.shape

        # since theta continuous, each particle (G, theta) is unique always
        logp = - jnp.log(N) * jnp.ones(N)

        return ParticleDistribution(logp=logp, g=g, theta=theta)


    def get_mixture(self, g, theta):
        """
        Converts batch of binary (adjacency) matrices and parameters into mixture particle distribution,
        where mixture weights correspond to unnormalized target (i.e. posterior) probabilities

        Args:
           g: [N, d, d] with {0,1} values
           theta: PyTree with leading dim `N`

        Returns:
           ParticleDistribution

        """
        N, _, _ = g.shape

        # mixture weighted by respective joint probabilities
        eltwise_log_joint_target = vmap(lambda single_g, single_theta:
                                        self.log_joint_prob(single_g, single_theta, self.x, None), (0, 0), 0)
        logp = eltwise_log_joint_target(g, theta)
        logp -= logsumexp(logp)

        return ParticleDistribution(logp=logp, g=g, theta=theta)
