import functools
import tqdm

import jax.numpy as jnp
from jax import jit, vmap, random, grad
from jax.experimental import optimizers

from dibs.inference.dibs import DiBS


class MarginalDiBS(DiBS):
    """
    This class implements DiBS: Differentiable Bayesian Structure Learning (Lorch et al., 2021)
    instantiated using Stein Variational Gradient Descent (SVGD) (Liu and Wang, 2016) as the underlying inference method.
    An SVGD update of vector v is defined as

        phi(v) = 1/n_particles sum_u k(v, u) d/du log p(u) + d/du k(u, v)

    This class implements //marginal// inference of p(G, theta | D).
    For joint inference of p(G | D), use the class `JointDiBS`

    Args:
        kernel:  object satisfying `BasicKernel` signature with differentiable evaluation function `eval()`
        target_log_prior: log p(G); differentiable log prior probability using the probabilities of edges (as implied by Z)
        target_log_marginal_prob: log p(D | G); differentiable or non-differentiable discrete log marginal probability of target distribution
        alpha_linear (float): inverse temperature parameter schedule of sigmoid
        beta_linear (float): inverse temperature parameter schedule of prior
        optimizer (dict): dictionary with at least keys `name` and `stepsize`
        n_grad_mc_samples (int): MC samples in gradient estimator for likelihood term p(theta, D | G)
        n_acyclicity_mc_samples (int): MC samples in gradient estimator for acyclicity constraint
        grad_estimator_z (str): gradient estimator d/dZ of expectation; choices: `score` or `reparam`
        score_function_baseline (float): weight of addition in score function baseline; == 0.0 corresponds to not using a baseline
        latent_prior_std (float): standard deviation of Gaussian prior over Z; defaults to 1/sqrt(k)
    """


    def __init__(self, *, kernel, target_log_prior, target_log_marginal_prob, alpha_linear, beta_linear=1.0, tau=1.0,
                 optimizer=dict(name='rmsprop', stepsize=0.005), n_grad_mc_samples=128, n_acyclicity_mc_samples=32, 
                 grad_estimator_z='score', score_function_baseline=0.0,
                 latent_prior_std=None, verbose=False):

        """
        To unify the function signatures for the marginal and joint inference classes `MarginalDiBS` and `JointDiBS`,
        we define a marginal log likelihood variant with dummy parameter inputs. This will allow using the same 
        gradient estimator functions for both inference cases.
        """
        target_log_marginal_prob_extra_args = lambda single_z, single_theta, rng: target_log_marginal_prob(single_z)

        super(MarginalDiBS, self).__init__(
            target_log_prior=target_log_prior,
            target_log_joint_prob=target_log_marginal_prob_extra_args,
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


    def sample_initial_random_particles(self, *, key, n_particles, n_vars, n_dim=None):
        """
        Samples random particles to initialize SVGD

        Args:
            key: rng key
            n_particles: number of particles for SVGD
            n_particles: number of variables `d` in inferred BN 
            n_dim: size of latent dimension `k`. Defaults to `n_vars`, s.t. k == d

        Returns:
            z: batch of latent tensors [n_particles, d, k, 2]    
        """
        # default full rank
        if n_dim is None:
            n_dim = n_vars 
        
        # like prior
        std = self.latent_prior_std or (1.0 / jnp.sqrt(n_dim))

        # sample
        key, subk = random.split(key)
        z = random.normal(subk, shape=(n_particles, n_vars, n_dim, 2)) * std        

        return z


    def f_kernel(self, x_latent, y_latent, h, t):
        """
        Evaluates kernel

        Args:
            x_latent: latent tensor [d, k, 2]
            y_latent: latent tensor [d, k, 2]
            h (float): kernel bandwidth 
            t: step

        Returns:
            [1, ] kernel value
        """
        return self.kernel.eval(x=x_latent, y=y_latent, h=h)
    

    def f_kernel_mat(self, x_latents, y_latents, h, t):
        """
        Computes pairwise kernel matrix

        Args:
            x_latents: latent tensor [A, d, k, 2]
            y_latents: latent tensor [B, d, k, 2]
            h (float): kernel bandwidth 
            t: step

        Returns:
            [A, B] kernel values
        """
        return vmap(vmap(self.f_kernel, (None, 0, None, None), 0), 
            (0, None, None, None), 0)(x_latents, y_latents, h, t)


    def eltwise_grad_kernel_z(self, x_latents, y_latent, h, t):
        """
        Computes gradient d/dz k(z, z') elementwise for each provided particle z

        Args:
            x_latents: batch of latent particles [n_particles, d, k, 2]
            y_latent: single latent particle [d, k, 2] (z')
            h (float): kernel bandwidth 
            t: step

        Returns:
            batch of gradients for latent tensors Z [n_particles, d, k, 2]
        """        
        grad_kernel_z = grad(self.f_kernel, 0)
        return vmap(grad_kernel_z, (0, None, None, None), 0)(x_latents, y_latent, h, t)


    def z_update(self, single_z, kxx, z, grad_log_prob_z, h, t):
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
        repulsion = self.eltwise_grad_kernel_z(z, single_z, h, t)

        # average and negate (for optimizer)
        return - (weighted_gradient_ascent + repulsion).mean(axis=0)


    def parallel_update_z(self, *args):
        """
        Parallelizes `z_update` for all available particles
        Otherwise, same inputs as `z_update`.
        """
        return vmap(self.z_update, (0, 1, None, None, None, None), 0)(*args)



    # this is the crucial @jit
    @functools.partial(jit, static_argnums=(0,))
    def svgd_step(self, opt_state_z, key, t, sf_baseline):
        """
        Performs a single SVGD step in the DiBS framework, updating all Z particles jointly.

        Args:
            opt_state_z: optimizer state for latent Z particles; contains [n_particles, d, k, 2]
            key: prng key
            t: step
            sf_baseline: batch of baseline values in case score function gradient is used [n_particles, ]

        Returns:
            the updated inputs
        """
     
        z = self.get_params(opt_state_z) # [n_particles, d, k, 2]
        n_particles = z.shape[0]

        # make sure same bandwith is used for all calls to k(x, x') (in case e.g. the median heuristic is applied)
        h = self.kernel.h

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
        kxx = self.f_kernel_mat(z, z, h, t)

        # transformation phi() applied in batch to each particle individually
        phi_z = self.parallel_update_z(z, kxx, z, dz_log_prob, h, t)

        # apply transformation
        # `x += stepsize * phi`; the phi returned is negated for SVGD
        opt_state_z = self.opt_update(t, phi_z, opt_state_z)

        return opt_state_z, key, sf_baseline
    
    

    def sample_particles(self, *, n_steps, init_particles_z, key, callback=None, callback_every=0):
        """
        Deterministically transforms particles to minimize KL to target using SVGD

        Arguments:
            n_steps (int): number of SVGD steps performed
            init_particles_z: batch of initialized latent tensor particles [n_particles, d, k, 2]
            key: prng key
            callback: function to be called every `callback_every` steps of SVGD.
            callback_every: if == 0, `callback` is never called. 

        Returns: 
            `n_particles` samples that approximate the DiBS target density
            particles_z: [n_particles, d, k, 2]
        """

        z = init_particles_z
           
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

        """Execute particle update steps for all particles in parallel using `vmap` functions"""
        it = tqdm.tqdm(range(n_steps), desc='DiBS', disable=not self.verbose)
        for t in it:

            # perform one SVGD step (compiled with @jit)
            opt_state_z, key, sf_baseline  = self.svgd_step(
                opt_state_z, key, t, sf_baseline)

            # callback
            if callback and callback_every and (((t+1) % callback_every == 0) or (t == (n_steps - 1))):
                z = self.get_params(opt_state_z)
                callback(
                    dibs=self,
                    t=t,
                    zs=z,
                )


        # return transported particles
        z_final = self.get_params(opt_state_z)
        return z_final
