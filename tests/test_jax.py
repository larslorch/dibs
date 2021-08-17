import os
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision

import collections
import pprint
import tqdm
import time
import numpy as onp

import jax.numpy as jnp
from jax import grad, jit, vmap, vjp, jvp
from jax import random
import jax.lax as lax
from jax.scipy.special import logsumexp

from dibs.kernel.basic import FrobeniusSquaredExponentialKernel

from dibs.utils.func import bit2id, id2bit

def egrad(f, x):
    '''Element-wise grad of f evaluated at x using explicit vector-Jacobian product'''
    fx, vjp_fun = vjp(f, x)
    ones = jnp.ones(x.shape[0], dtype=jnp.float32)
    out = vjp_fun(ones)[0]
    return out


if __name__ == '__main__':

    jnp.set_printoptions(precision=4, suppress=True)

    # Generate key which is used to generate random numbers
    key = random.PRNGKey(0)

    '''
    Pytree map
    '''

    from jax.tree_util import tree_map

    a = jnp.array([[1, 2], [10, 20]], dtype=jnp.float32)
    b = jnp.array([[3, 4, 5], [30, 40, 50]], dtype=jnp.float32)
    c = jnp.array([[5, 6]], dtype=jnp.float32)

    pytr = [a, b, c]
    print(pytr)

    g = 1.23456

    def __logprob__(params):
        return jnp.square(params - g).sum()

    @jit
    def logprob_params(theta):
        return jnp.array(tree_map(__logprob__, theta)).sum()

    grad_logprob_params = grad(logprob_params, 0)

    res = logprob_params(pytr)
    print(res)

    dtheta_res = grad_logprob_params(pytr)
    print(dtheta_res)
    print()

    '''
    bit2id
    '''
    n_tests = 100
    N = 5

    for n_vars in [5, 8, 10, 20, 200]:

        print('n_vars', n_vars)

        for t in range(n_tests):

            key, subk = random.split(key)
            b = random.bernoulli(subk, p=0.5, shape=(N, n_vars, n_vars)).astype(jnp.int64)
            if t == 0:
                b = jnp.zeros((N, n_vars, n_vars)).astype(jnp.int64)
            elif t == 1:
                b = jnp.ones((N, n_vars, n_vars)).astype(jnp.int64)

            id = bit2id(b)
            b2 = id2bit(id, n_vars)
            assert(jnp.allclose(b, b2))

    print('bit2id done.')

    '''
    Squared Norm of Gaussian
    '''
    n = 500

    for k in range(100):
        key, subk = random.split(key)
        w = random.normal(subk, (n, k)) / jnp.sqrt(k)
        squared_norm = jnp.einsum('nk,nk->n', w, w)
        # print('k = {}: {}'.format(k, squared_norm.mean(0)))

    for k in range(100):
        key, subk = random.split(key)
        u = random.normal(subk, (n, k)) / jnp.sqrt(k)
        key, subk = random.split(key)
        w = random.normal(subk, (n, k)) / jnp.sqrt(k)
        d = u - w
        squared_norm = jnp.einsum('nk,nk->n', d, d)
        # print('k = {}: {}'.format(k, squared_norm.mean(0)))


    '''
    Gradient logsumexp
    '''

    mc_numerator = 50
    mc_denominator = 20
    latent_dim = 3 * 3
    provoke_nans = False

    # possible different no. mc samples for either computatin
    # [mc_numerator, ]
    logprobs_numerator = random.uniform(key, shape=(mc_numerator, ), 
        minval=-100 if provoke_nans else -5, 
        maxval=-90 if provoke_nans else -1)
    probs_numerator = jnp.exp(logprobs_numerator)

    # [mc_denominator, ]
    logprobs_denominator = random.uniform(key, shape=(mc_denominator, ), 
        minval=-100 if provoke_nans else -5, 
        maxval=-90 if provoke_nans else -1)    
    probs_denominator = jnp.exp(logprobs_denominator)

    # [latent_dim, mc_numerator]
    grad_x = random.normal(key, shape=(latent_dim, mc_numerator))

    ########## naive
    # [latent_dim, ]
    numerator = jnp.mean(probs_numerator * grad_x, axis=1)

    # []
    denominator = jnp.mean(probs_denominator, axis=0)

    # score function grad
    # [latent_dim, ]
    sf_grad = numerator / denominator

    ########## stable
    # [latent_dim, ]  [latent_dim, ]
    log_numerator, sign = logsumexp(a=logprobs_numerator, b=grad_x, axis=1, return_sign=True)

    # [latent_dim, ]
    stable_numerator = sign * jnp.exp(log_numerator - jnp.log(mc_numerator))

    # []
    log_denominator = logsumexp(logprobs_denominator, axis=0)
    stable_denominator = jnp.exp(log_denominator - jnp.log(mc_denominator))

    # stable score function grad
    # [latent_dim, ]
    stable_sf_grad = sign * jnp.exp(log_numerator - jnp.log(mc_numerator) - log_denominator + jnp.log(mc_denominator))

    print(jnp.allclose(sf_grad, stable_sf_grad))
    print()


    '''
    Avoiding np.where for marginal likelihoods
    '''

    key, subk = random.split(key)
    graphs = random.randint(subk, (10, 5, 5), 0, 2)
    print(graphs.shape)

    key, subk = random.split(key)
    x = random.normal(subk, (4, 5))

    print('x   \n', x, end='\n\n', flush=True)

    eye = jnp.eye(5)
    ones = jnp.ones((5, 5))
    zeros = jnp.zeros((5, 5))


    xpad = jnp.concatenate([x, jnp.zeros((x.shape[0], 4))], axis=1)

    # print(x @ x.T)
    # print(xpad @ xpad.T)



    @jit
    def sel(mat, mask):
        '''
            jit/vmap helper function

            mat:   [N, d]
            mask:  [d, ]   boolean 

            returns [N, d] with columns of `mat` with `mask` == 1 non-zero a
            and pushed leftmost; the columns with `mask` == 0 are zero

            e.g. 
            mat 
            1 2 3
            4 5 6
            7 8 9

            mask
            1 0 1

            out
            1 3 0
            4 6 0
            7 9 0
        '''
        valid_indices = jnp.where(mask, jnp.arange(mask.shape[0]), mask.shape[0])
        padded_mat = jnp.concatenate([mat, jnp.zeros((mat.shape[0], 1))], axis=1)
        padded_valid_mat = padded_mat[:, jnp.sort(valid_indices)]
        return padded_valid_mat



    def f(w):
        logprob = 0
        for j in range(w.shape[-1]):
            
            ### works
            # parents = jnp.where(w[:, j] == 1, eye, zeros)
            # val = (x @ parents).sum()
            # logprob += val

            ### doesnt work
            # parents = jnp.where(w[:, j] == 1, 1, 0)
            # val = jnp.compress(parents, x, axis=1).sum()
            # logprob += val

            # works
            parents = w[:, j] == 1
            n_parents = jnp.sum(parents)
            x_pa_pad = sel(x, parents)
            logprob += x_pa_pad.sum()

        return logprob

    fv = vmap(f, 0, 0)
    

    print(f(graphs[0]))
    print(fv(graphs))
    print()







    '''Understanding JAX'''

    '''   +++ vmap +++
    
    vmap(f, in_axes, out_axes): 

    length of `in_axes` tuple === number of positional arguments of f
        e.g. if f = lambda x, y: x + y
             then len(in_axes) == 2

    the tuple `in_axes` indicates which axis to map for each positional argument of f,
    where to map means to iteratively apply the function over (in the spirit of `for i in range(...)`) in a parallelized fashion
        e.g. (None, 0, 0) means “don't parallelize over the first argument, and parallelize over the 0-th dimension of the second and third arguments".
        if in_axis[i] is an integer k, then the k-th axis of the positional argument will be mapped over
        if in_axis[i] is `None`, then the axis is not mapped
    
    `out_axis` indicates where the mapped axis should be in the output
    '''

    print('vmap')
    key, *subkey = random.split(key, 3)

    a = random.normal(subkey[0], (10,))
    b = random.normal(subkey[1], (10,))

    key, *subkey = random.split(key, 3)

    M = random.normal(subkey[0], (4, 10))
    N = random.normal(subkey[1], (10, 6))

    # vector vector multiplication
    vv = lambda x, y: jnp.vdot(x, y)  #  ([a], [a]) -> []
    print('vv', vv(a, b))

    # matrix-vector multiplication: equivalent ways of doing the same thing
    print('mv   ', vmap(vv, (0, None), 0)(M, a))   # ([b,a], [a])    -> [b]  (0-th axis of 0-th argument is mapped over)
    print('mv   ', vmap(vv, (None, 0), 0)(a, M))   # ([a],   [b,a])  -> [b]  (0-th axis of 1-st argument is mapped over)
    print('mv   ', vmap(vv, (1, None), 0)(M.T, a)) # ([a,b], [a])    -> [b]  (1-st axis of 0-th argument is mapped over)
    print('mv   ', vmap(vv, (None, 1), 0)(a, M.T)) # ([a],   [a,b])  -> [b]  (1-st axis of 1-st argument is mapped over)
    print('numpy', jnp.matmul(M, a))
    print()


    # matrix-matrix multiplication: equivalent ways of doing the same thing
    mv = vmap(vv, (0, None), 0)  # ([b,a], [a])    -> [b]
    print('mm    \n', vmap(mv, (None, 1), 1)(M, N))  # ([b,a], [a,c]) -> [b,c]  (1-st axis of 1-st argument is mapped over, i.e. c; and put in 1-st axis of output)

    mv = vmap(vv, (None, 0), 0)  # ([a],   [b,a])  -> [b]
    print('mm    \n', vmap(mv, (1, None), 1)(N, M))  # ([a,c], [b,a]) -> [b,c]  (1-st axis of 0-th argument is mapped over, i.e. c; and put in 1-st axis of output)
    print()


    '''jit'''
    print('jit')

    n_particles, n_dim = 30, 2

    key, *subkey = random.split(key, 4)

    all_x = random.normal(subkey[0], (n_particles, n_dim)) # [30, 2]
    grad_log_prob = random.normal(subkey[1], (n_particles, n_dim))  # [30, 2]

    kernel = FrobeniusSquaredExponentialKernel()  # h not set indicates median trick

    def f_kernel(a, b): 
        return kernel.eval(x=a, y=b)
    
    kxx = f_kernel(all_x, all_x)
    k = jit(f_kernel)
    kxx_fast = k(all_x, all_x)

    print()


    '''kxx'''
    print('vmap svgd transform')

    # vmap does not recognize computation of h in parallel, so it would use
    # different h for each particle unless we compute it beforehand
    h_forall = lax.stop_gradient(
        kernel.compute_median_heuristic(x=all_x, y=all_x))

    # assuming x and kxx_for_x have all particles in 0-th dimension, later want to parallelize
    def single_x_update(x, kxx_for_x):

        # compute all sum terms
        gradient_ascent = kxx_for_x[:, jnp.newaxis] * grad_log_prob # [30, 2] = [30, 1] * [30, 2]
        repulsion = egrad(lambda a: kernel.eval(x=a, y=x, h=h_forall), all_x)  # [30, 2]

        return (gradient_ascent + repulsion).mean(axis=0)

    a = single_x_update(all_x[0], kxx[:, 0])
    f = vmap(single_x_update, (0, 1), 0)
    print(f(all_x, kxx).shape)
    print()

   
    '''grad'''

    print('grad', flush=True)

    kernel = FrobeniusSquaredExponentialKernel()  # h not set indicates median trick

    # make sure same bandwith is used for all calls to k(x,x') if the median heuristic is applied
    h_forall = lax.stop_gradient(kernel.compute_median_heuristic(x=all_x, y=all_x)) 
    print('h_forall  = ', h_forall, flush=True)

    f = lambda a, b, h: kernel.eval(x=a, y=b, h=h)
    grad_f = grad(f, 0) 
    grad_f_b = vmap(grad_f, (0, None, None), 0)

    t0 = time.time()
    eltwise_grad_vmap = grad_f_b(all_x, all_x[1], h_forall)
    t1 = time.time()
    eltwise_grad_vjp = egrad(lambda a: kernel.eval(x=a, y=all_x[1], h=h_forall), all_x)
    t2 = time.time()

    print('vmap: {:.04f}'.format(t1 - t0))
    print('vjp:  {:.04f}'.format(t2 - t1))

    print(jnp.allclose(eltwise_grad_vmap, eltwise_grad_vjp))

   
