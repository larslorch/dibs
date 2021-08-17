import functools
import tqdm

import jax.numpy as jnp
from jax import jit, vmap, random, grad
from jax.experimental import optimizers
from jax.tree_util import tree_map, tree_multimap

from dibs.inference.dibs import DiBS

from dibs.utils.func import expand_by


class JointDiBS(DiBS):
    """
    This class implements DiBS: Differentiable Bayesian Structure Learning (Lorch et al., 2021)
    instantiated using Stein Variational Gradient Descent (SVGD) (Liu and Wang, 2016) as the underlying inference method.
    An SVGD update of vector v is defined as

        phi(v) = 1/n_particles sum_u k(v, u) d/du log p(u) + d/du k(u, v)
        
    This class implements //joint// inference of the posterior p(G, theta | D).
    For marginal inference of p(G | D), use the class `MarginalDiBS`

    Args:
        kernel:  object satisfying `BasicKernel` signature with differentiable evaluation function `eval()`
        target_log_prior: log p(G); differentiable log prior probability using the probabilities of edges (as implied by Z)
        target_log_joint_prob: log p(theta, D | G); differentiable or non-differentiable discrete log probability of target distribution
        alpha_linear (float): inverse temperature parameter schedule of sigmoid
        beta_linear (float): inverse temperature parameter schedule of prior
        optimizer (dict): dictionary with at least keys `name` and `stepsize`
        n_grad_mc_samples (int): MC samples in gradient estimator for likelihood term p(theta, D | G)
        n_acyclicity_mc_samples (int): MC samples in gradient estimator for acyclicity constraint
        grad_estimator_z (str): gradient estimator d/dZ of expectation; choices: `score` or `reparam`
        score_function_baseline (float): weight of addition in score function baseline; == 0.0 corresponds to not using a baseline
        latent_prior_std (float): standard deviation of Gaussian prior over Z; defaults to 1/sqrt(k)

    """

    def __init__(self, *, kernel, target_log_prior, target_log_joint_prob, alpha_linear, beta_linear=1.0, tau=1.0,
                 optimizer=dict(name='rmsprop', stepsize=0.005), n_grad_mc_samples=128, n_acyclicity_mc_samples=32, 
                 grad_estimator_z='reparam', score_function_baseline=0.0,
                 latent_prior_std=None, verbose=False):
        super(JointDiBS, self).__init__(
            target_log_prior=target_log_prior,
            target_log_joint_prob=target_log_joint_prob,
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

        self.kernel = kernel
        self.optimizer = optimizer
       

    def sample_initial_random_particles(self, *, key, n_particles, n_vars, model, n_dim=None):
        """
        Samples random particles to initialize SVGD

        Args:
            key: rng key
            n_particles: number of particles for SVGD
            n_particles: number of variables `d` in inferred BN 
            model: model of type `BasicModel` to sample prior parameter PyTree
            n_dim: size of latent dimension `k`. Defaults to `n_vars`, s.t. k == d

        Returns:
            z: batch of latent tensors [n_particles, d, k, 2]
            theta: batch of parameters PyTree with leading dim `n_particles`
        
        """
        # default full rank
        if n_dim is None:
            n_dim = n_vars 
        
        # std like Gaussian prior over Z           
        std = self.latent_prior_std  or (1.0 / jnp.sqrt(n_dim))

        # sample
        key, subk = random.split(key)
        z = random.normal(subk, shape=(n_particles, n_vars, n_dim, 2)) * std

        key, subk = random.split(key)
        theta = model.init_parameters(key=subk, n_particles=n_particles, n_vars=n_vars)

        return z, theta


    def f_kernel(self, x_latent, x_theta, y_latent, y_theta, h_latent, h_theta, t):
        """
        Evaluates kernel

        Args:
            x_latent: latent tensor [d, k, 2]
            x_theta: parameter PyTree 
            y_latent: latent tensor [d, k, 2]
            y_theta: parameter PyTree 
            h_latent (float): kernel bandwidth for Z term
            h_theta (float): kernel bandwidth for theta term
            t: step

        Returns:
            kernel value
        """
        return self.kernel.eval(
            x_latent=x_latent, x_theta=x_theta,
            y_latent=y_latent, y_theta=y_theta,
            h_latent=h_latent, h_theta=h_theta)
    

    def f_kernel_mat(self, x_latents, x_thetas, y_latents, y_thetas, h_latent, h_theta, t):
        """
        Computes pairwise kernel matrix

        Args:
            x_latents: latent tensor [A, d, k, 2]
            x_thetas: parameter PyTree with batch size A as leading dim
            y_latents: latent tensor [B, d, k, 2]
            y_thetas: parameter PyTree with batch size B as leading dim
            h_latent (float): kernel bandwidth for Z term
            h_theta (float): kernel bandwidth for theta term
            t: step

        Returns:
            [A, B] kernel values
        """
        return vmap(vmap(self.f_kernel, (None, None, 0, 0, None, None, None), 0), 
            (0, 0, None, None, None, None, None), 0)(x_latents, x_thetas, y_latents, y_thetas, h_latent, h_theta, t)

    
    def eltwise_grad_kernel_z(self, x_latents, x_thetas, y_latent, y_theta, h_latent, h_theta, t):
        """
        Computes gradient d/dz k((z, theta), (z', theta')) elementwise for each provided particle (z, theta)

        Args:
            x_latents: batch of latent particles [n_particles, d, k, 2]
            x_thetas: batch of parameter PyTree with leading dim `n_particles`
            y_latent: single latent particle [d, k, 2] (z')
            y_theta: single parameter PyTree (theta')
            h_latent (float): kernel bandwidth for Z term
            h_theta (float): kernel bandwidth for theta term
            t: step

        Returns:
            batch of gradients for latent tensors Z [n_particles, d, k, 2]
        
        """        
        grad_kernel_z = grad(self.f_kernel, 0)
        return vmap(grad_kernel_z, (0, 0, None, None, None, None, None), 0)(x_latents, x_thetas, y_latent, y_theta, h_latent, h_theta, t)


    def eltwise_grad_kernel_theta(self, x_latents, x_thetas, y_latent, y_theta, h_latent, h_theta, t):
        """
        Computes gradient d/dtheta k((z, theta), (z', theta')) elementwise for each provided particle (z, theta)

        Args:
            x_latents: batch of latent particles [n_particles, d, k, 2]
            x_thetas: batch of parameter PyTree with leading dim `n_particles`
            y_latent: single latent particle [d, k, 2] (z')
            y_theta: single parameter PyTree (theta')
            h_latent (float): kernel bandwidth for Z term
            h_theta (float): kernel bandwidth for theta term
            t: step

        Returns:
            batch of gradients for parameters (PyTree with leading dim `n_particles`)
        """
        grad_kernel_theta = grad(self.f_kernel, 1)
        return vmap(grad_kernel_theta, (0, 0, None, None, None, None, None), 0)(x_latents, x_thetas, y_latent, y_theta, h_latent, h_theta, t)


    def z_update(self, single_z, single_theta, kxx, z, theta, grad_log_prob_z, h_latent, h_theta, t):
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
        repulsion = self.eltwise_grad_kernel_z(z, theta, single_z, single_theta, h_latent, h_theta, t)

        # average and negate (for optimizer)
        return - (weighted_gradient_ascent + repulsion).mean(axis=0)


    def parallel_update_z(self, *args):
        """
        Parallelizes `z_update` for all available particles
        Otherwise, same inputs as `z_update`.
        """
        return vmap(self.z_update, (0, 0, 1, None, None, None, None, None, None), 0)(*args)


    def theta_update(self, single_z, single_theta, kxx, z, theta, grad_log_prob_theta, h_latent, h_theta, t):
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
        
        repulsion = self.eltwise_grad_kernel_theta(z, theta, single_z, single_theta, h_latent, h_theta, t)

        # average and negate (for optimizer)
        return  tree_multimap(
            lambda grad_asc_leaf, repuls_leaf: 
                - (grad_asc_leaf + repuls_leaf).mean(axis=0), 
            weighted_gradient_ascent, 
            repulsion)


    def parallel_update_theta(self, *args):
        """
        Parallelizes `theta_update` for all available particles
        Otherwise, same inputs as `theta_update`.
        """
        return vmap(self.theta_update, (0, 0, 1, None, None, None, None, None, None), 0)(*args)


    # this is the crucial @jit
    @functools.partial(jit, static_argnums=(0,))
    def svgd_step(self, opt_state_z, opt_state_theta, key, t, sf_baseline):
        """
        Performs a single SVGD step in the DiBS framework, updating Z and theta jointly.
        
        Args:
            opt_state_z: optimizer state for latent Z particles; contains [n_particles, d, k, 2]
            opt_state_theta: optimizer state for theta particles; contains PyTree with `n_particles` leading dim
            key: prng key
            t: step
            sf_baseline: batch of baseline values in case score function gradient is used [n_particles, ]

        Returns:
            the updated inputs
        """
     
        z = self.get_params(opt_state_z) # [n_particles, d, k, 2]
        theta = self.get_params(opt_state_theta) # PyTree with `n_particles` leading dim
        n_particles = z.shape[0]

        # make sure same bandwith is used for all calls to k(x, x') (in case e.g. the median heuristic is applied)
        h_latent = self.kernel.h_latent
        h_theta = self.kernel.h_theta

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
        kxx = self.f_kernel_mat(z, theta, z, theta, h_latent, h_theta, t)

        # transformation phi() applied in batch to each particle individually
        phi_z = self.parallel_update_z(z, theta, kxx, z, theta, dz_log_prob, h_latent, h_theta, t)
        phi_theta = self.parallel_update_theta(z, theta, kxx, z, theta, dtheta_log_prob, h_latent, h_theta, t)

        # apply transformation
        # `x += stepsize * phi`; the phi returned is negated for SVGD
        opt_state_z = self.opt_update(t, phi_z, opt_state_z)
        opt_state_theta = self.opt_update(t, phi_theta, opt_state_theta)

        return opt_state_z, opt_state_theta, key, sf_baseline
    
    

    def sample_particles(self, *, n_steps, init_particles_z, init_particles_theta, key, callback=None, callback_every=0):
        """
        Deterministically transforms particles to minimize KL to target using SVGD

        Arguments:
            n_steps (int): number of SVGD steps performed
            init_particles_z: batch of initialized latent tensor particles [n_particles, d, k, 2]
            init_particles_theta:  batch of parameters PyTree (i.e. for a general parameter set shape) 
                with leading dimension `n_particles`
            key: prng key
            callback: function to be called every `callback_every` steps of SVGD.
            callback_every: if == 0, `callback` is never called. 

        Returns: 
            `n_particles` samples that approximate the DiBS target density
            particles_z: [n_particles, d, k, 2]
            particles_theta: PyTree of parameters with leading dimension `n_particles`
           
        """

        z = init_particles_z
        theta = init_particles_theta
           
        # initialize score function baseline (one for each particle)
        n_particles, _, n_dim, _ = z.shape
        sf_baseline = jnp.zeros(n_particles)

        if self.latent_prior_std is None:
            self.latent_prior_std = 1.0 / jnp.sqrt(n_dim)


        # init optimizer
        if self.optimizer['name'] == 'gd':
            opt = optimizers.sgd(self.optimizer['stepsize']/ 10.0) # comparable scale for tuning
        elif self.optimizer['name'] == 'momentum':
            opt = optimizers.momentum(self.optimizer['stepsize'])
        elif self.optimizer['name'] == 'adagrad':
            opt = optimizers.adagrad(self.optimizer['stepsize'])
        elif self.optimizer['name'] == 'adam':
            opt = optimizers.adam(self.optimizer['stepsize'])
        elif self.optimizer['name'] == 'rmsprop':
            opt = optimizers.rmsprop(self.optimizer['stepsize'])
        else:
            raise ValueError()
        
        opt_init, self.opt_update, get_params = opt
        self.get_params = jit(get_params)
        opt_state_z = opt_init(z)
        opt_state_theta = opt_init(theta)

        """Execute particle update steps for all particles in parallel using `vmap` functions"""
        it = tqdm.tqdm(range(n_steps), desc='DiBS', disable=not self.verbose)
        for t in it:

            # perform one SVGD step (compiled with @jit)
            opt_state_z, opt_state_theta, key, sf_baseline  = self.svgd_step(
                opt_state_z, opt_state_theta, key, t, sf_baseline)

            # callback
            if callback and callback_every and (((t+1) % callback_every == 0) or (t == (n_steps - 1))):
                z = self.get_params(opt_state_z)
                theta = self.get_params(opt_state_theta)
                callback(
                    dibs=self,
                    t=t,
                    zs=z,
                    thetas=theta,
                )


        # return transported particles
        z_final = self.get_params(opt_state_z)
        theta_final = self.get_params(opt_state_theta)
        return z_final, theta_final
