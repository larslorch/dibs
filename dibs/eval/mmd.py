import jax.numpy as jnp
import jax.lax as lax
from jax.scipy.special import logsumexp
from dibs.utils.func import id2bit

class MaximumMeanDiscrepancy:
    """
    Computes unbiased MMD estimate given two graph sample sets of sizes M and N, respectively
    as in Eq. 3 of Gretton et al, 2008
    
    https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf

    """

    def __init__(self, *, kernel):
        super(MaximumMeanDiscrepancy, self).__init__()

        self.kernel = kernel

    def jit_squared_mmd(self, *, p_samples, q_samples, mmd_h=-1.0):
        """Computes unbiased MMD estimate
            p_samples:  [N, ...], N samples from a distribution with arbitrary dimensionality (...) 
            q_samples:  [M, ...], M samples from a distribution with arbitrary dimensionality (...) 
        
        arbitrary dimensionality means that a single sample, e.g. p_samples[i] can be a vector, matrix, or any other tensor (any np.ndim)
        """
        N, M = p_samples.shape[0], q_samples.shape[0]

        # bandwidth
        h_ = lax.cond(
            mmd_h == -1.0,
            lambda _: self.kernel.h,
            lambda _: mmd_h,
            operand=None)

        h__ = lax.cond(
            h_ == -1.0,
            lambda _: self.kernel.compute_median_heuristic(
                x=jnp.concatenate([p_samples, q_samples], axis=0),
                y=jnp.concatenate([p_samples, q_samples], axis=0)),
            lambda _: h_,
            operand=None)
        
        # compute kernels
        kernel_xy = self.kernel.eval(x=p_samples, y=q_samples, h=h__) / (N * M)

        # if more than 1 particle
        kernel_xx = lax.cond(N > 1, 
            lambda _: self.kernel.eval(x=p_samples, y=p_samples, h=h__) / (N * (N - 1)),
            lambda _: jnp.zeros((N, N)), operand=None)
        
        kernel_yy = lax.cond(M > 1,
            lambda _: self.kernel.eval(x=q_samples, y=q_samples, h=h__) / (M * (M - 1)),
            lambda _: jnp.zeros((M, M)), operand=None)
        
        mmd = kernel_xx.sum() + kernel_yy.sum() - 2 * kernel_xy.sum()

        # ignore diags of xx and yy (if more than 1 particles)
        mmd -= lax.cond(N > 1, 
            lambda _: kernel_xx[jnp.diag_indices(N)].sum(),
            lambda _: 0.0, operand=None)

        mmd -= lax.cond(M > 1, 
            lambda _: kernel_yy[jnp.diag_indices(M)].sum(),
            lambda _: 0.0, operand=None)

        return mmd

    def __expectation_sample_sample(self, *, samples_a, samples_b, h, same_samples):
        """Estimates/computes expectation term in MMD for distribution `a` and `b`
            for 2 sets of samples
        """

        N, M = samples_a.shape[0], samples_b.shape[0]
        
        # compute kernels using U-statistic for `samples_a` and `samples_b`
        kernel_ab = self.kernel.eval(x=samples_a, y=samples_b, h=h)
        if same_samples:
            assert(N == M)
            return (jnp.tril(kernel_ab, k=-1).sum() + jnp.triu(kernel_ab, k=1).sum()) / (N * (M - 1))
        else:
            return kernel_ab.sum() / (N * M)


    def __expectation_dist_sample(self, *, dist_a, samples_b, h):
        """Estimates/computes expectation term in MMD for distribution `a` and `b`
            for pair of 1 set of sample and 1 tractable distribution
        """

        N_b, n_vars, _ = samples_b.shape

        dist_a_id_particles, dist_a_log_weights = dist_a
        dist_a_particles = id2bit(dist_a_id_particles, n_vars)

        # compute kernels using exact expectation for `dist_a` and U-statistic for `samples_b`
        kernel_ab = self.kernel.eval(x=dist_a_particles, y=samples_b, h=h)
        kernel_a = kernel_ab.sum(axis=1)

        # expectation = jnp.sum(jnp.exp(dist_a_log_weights) * kernel_a) / N_b
        log_expectation, log_expectation_sgn = logsumexp(dist_a_log_weights, b=kernel_a, return_sign=True)
        expectation = log_expectation_sgn * jnp.exp(log_expectation - jnp.log(N_b))
        return expectation


    def __expectation_dist_dist(self, *, dist_a, dist_b, h, n_vars):
        """Estimates/computes expectation term in MMD for distribution `a` and `b`
            for pair of 2 tractable distributions
        """

        dist_a_id_particles, dist_a_log_weights = dist_a
        dist_a_particles = id2bit(dist_a_id_particles, n_vars)

        dist_b_id_particles, dist_b_log_weights = dist_b
        dist_b_particles = id2bit(dist_b_id_particles, n_vars)

        # compute kernels using exact expectation for `dist_a` and `dist_b`
        kernel_ab = self.kernel.eval(x=dist_a_particles, y=dist_b_particles, h=h)
        log_weight_products = dist_a_log_weights[:, None] + dist_b_log_weights[None]

        # expectation = jnp.sum(jnp.exp(log_weight_products) * kernel_ab)
        log_expectation, log_expectation_sgn = logsumexp(log_weight_products, b=kernel_ab, return_sign=True)
        expectation = log_expectation_sgn * jnp.exp(log_expectation)

        return expectation


    def squared_mmd(self, *, p, q, mmd_h, n_vars):
        """Computes unbiased MMD estimate
            p:  [N, ...], N samples from a distribution with arbitrary dimensionality (...) 
                or distribution tuple
            q:  [M, ...], M samples from a distribution with arbitrary dimensionality (...) 
                or distribution tuple

        arbitrary dimensionality means that a single sample, e.g. p_samples[i] can be a vector, matrix, or any other tensor (any np.ndim)
        """

        p_is_dist = isinstance(p, tuple)
        q_is_dist = isinstance(q, tuple)
        if not p_is_dist and not q_is_dist:
            if p.shape[0] == 1 or q.shape[0] == 1:
                raise ValueError('Degenerate samples in MaximumMeanDiscrepancy.squared_mmd: p and q only consist of 1 particle')

        mmd = 0.0

        # E_pp k(x, x)
        mmd += (self.__expectation_dist_dist(dist_a=p, dist_b=p, h=mmd_h, n_vars=n_vars) if p_is_dist else
                self.__expectation_sample_sample(samples_a=p, samples_b=p, h=mmd_h, same_samples=True))

        # E_qq k(y, y)
        mmd += (self.__expectation_dist_dist(dist_a=q, dist_b=q, h=mmd_h, n_vars=n_vars) if q_is_dist else
                self.__expectation_sample_sample(samples_a=q, samples_b=q, h=mmd_h, same_samples=True))

        # - 2 * E_pq k(x, y)
        if p_is_dist and q_is_dist:
            mmd -= 2.0 * self.__expectation_dist_dist(dist_a=p, dist_b=q, h=mmd_h, n_vars=n_vars)
        elif p_is_dist and not q_is_dist:
            mmd -= 2.0 * self.__expectation_dist_sample(dist_a=p, samples_b=q, h=mmd_h)
        elif not p_is_dist and q_is_dist:
            mmd -= 2.0 * self.__expectation_dist_sample(dist_a=q, samples_b=p, h=mmd_h)
        elif not p_is_dist and not q_is_dist:
            mmd -= 2.0 * self.__expectation_sample_sample(samples_a=p, samples_b=q, h=mmd_h, same_samples=False)
        else:
            raise ValueError('Fatal error in MaximumMeanDiscrepancy.squared_mmd')

        return mmd 
