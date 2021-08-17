import os
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision

import collections
import pprint
import tqdm
import time
import numpy as onp

import jax.numpy as jnp
from jax import grad, jit, vmap, vjp, jvp, jacfwd, jacrev
from jax import random
import jax.lax as lax
from jax.scipy.special import logsumexp

from dibs.kernel.basic import FrobeniusSquaredExponentialKernel

from dibs.utils.func import bit2id, id2bit

from dibs.utils.graph import acyclic_constr, acyclic_constr_nograd

if __name__ == '__main__':

    jnp.set_printoptions(precision=4, suppress=True)
    key = random.PRNGKey(0)

    # n_vars, n_dim = 40, 15
    # n_vars, n_dim = 30, 10
    n_vars, n_dim = 20, 10
    # n_vars, n_dim = 10, 5
    # n_vars, n_dim = 5, 3

    alpha = 1.0

    # forward functions
    def Z_to_G(Z, U):

        scores = jnp.einsum('ik,jk->ij', Z[:, :, 0], Z[:, :, 1])
        # probs = 1 / (1 + jnp.exp(- alpha * scores))

        # soft reparameterization using gumbel
        soft = 1 / (1 + jnp.exp(- (jnp.log(U) - jnp.log(1 - U) + alpha * scores)))

        # sample
        # hard = (soft > 0.5).astype(jnp.float64)

        return soft
        # return hard

    def constraint(Z, U):
        
        # Z -> G
        G = Z_to_G(Z, U)
        
        # G -> h
        h = acyclic_constr_nograd(G, n_vars)
        return h

    # grad functions
    # jacobian Z -> G
    jac_Z_to_G = jacrev(Z_to_G, 0)

    # grad G -> h
    grad_acyclic_constr = grad(acyclic_constr_nograd, 0)

    def grad_constraint(Z, U):
        '''
        Pathwise derivative of d/dZ h(G(Z, U))
        Allows for using hard G|Z samples in forward pass
        and Gumbel softmax in backward pass
        '''

        # dG(Z)/dZ: [d, d, d, k, 2]
        # since mapping [d, k, 2] -> [d, d]
        jac_G_Z = jac_Z_to_G(Z, U)

        # G(Z)
        G = Z_to_G(Z, U) 

        # dh(G)/dG
        grad_h_G = grad_acyclic_constr(G, n_vars)

        # pathwise derivative
        # sum over all i,j in G (from h back to Z)
        dZ = jnp.sum(grad_h_G[:, :, None, None, None] * jac_G_Z, axis=(0, 1))
        return dZ


    
    key, subk = random.split(key)
    Z = random.normal(subk, shape=(n_vars, n_dim, 2))
    print('Z', Z.shape)

    key, subk = random.split(key)
    U = random.uniform(subk, shape=(n_vars, n_vars))
    print('U', U.shape)

    key, subk = random.split(key)
    h = constraint(Z, U)
    print('h', h)

    # use same key here onward
    '''
    Autodiff
    '''
    grad_constraint_auto = grad(constraint, 0) 

    t0 = time.time()
    dZ_auto = grad_constraint_auto(Z, U)
    # print(dZ_auto.shape)
    t1 = time.time()
    print(t1 - t0)

    '''
    Own diff
    (same key)
    '''    
    t0 = time.time()
    dZ = grad_constraint(Z, U)
    # print(dZ.shape)
    t1 = time.time()
    print(t1 - t0)

    print('match: ', jnp.allclose(dZ_auto, dZ))


    '''
    Comparison of gradient variants
    '''

    print('comparison of soft/hard sample/prob grad')
    def grad_constraint_hard(Z, U):
        '''
        Pathwise derivative of d/dZ h(G(Z, U))
        Allows for using hard G|Z samples in forward pass
        and Gumbel softmax in backward pass
        '''

        # dG/dZ: [d, d, d, k, 2]
        # since mapping [d, k, 2] -> [d, d]
        jac_G_Z = jac_Z_to_G(Z, U)

        G = (Z_to_G(Z, U) > 0.5).astype(jnp.float64)

        # dh/dG
        grad_h_G = grad_acyclic_constr(G, n_vars)

        # pathwise derivative
        # sum over all i,j in G (from h back to Z)
        dZ = jnp.sum(grad_h_G[:, :, None, None, None] * jac_G_Z, axis=(0, 1))
        return dZ
    
    def constraint_prob(Z):
        '''
        Current version
        '''
        scores = jnp.einsum('ik,jk->ij', Z[:, :, 0], Z[:, :, 1])
        probs = 1 / (1 + jnp.exp(- alpha * scores))
        h = acyclic_constr_nograd(probs, n_vars)
        return h

    grad_constraint_prob = jit(grad(constraint_prob, 0))

    mc_samples = 100
    key, subk = random.split(key)
    U = random.uniform(subk, shape=(mc_samples, n_vars, n_vars))
    print('U', U.shape)

    batch_grad_constraint_auto = jit(vmap(grad_constraint_auto, (None, 0), 0))
    batch_grad_constraint = jit(vmap(grad_constraint, (None, 0), 0))
    batch_grad_constraint_hard = jit(vmap(grad_constraint_hard, (None, 0), 0))

    t0 = time.time()
    auto_grad = batch_grad_constraint_auto(Z, U).mean(0)
    t1 = time.time()
    print('auto soft', t1 - t0)

    t0 = time.time()
    soft_grad = batch_grad_constraint(Z, U).mean(0)
    t1 = time.time()
    print('soft', t1 - t0)

    t0 = time.time()
    hard_grad = batch_grad_constraint_hard(Z, U).mean(0)
    t1 = time.time()
    print('hard', t1 - t0)

    t0 = time.time()
    prob_constraint = grad_constraint_prob(Z)
    t1 = time.time()
    print('prob', t1 - t0)

    print(soft_grad.shape)
    print(hard_grad.shape)
    print(prob_constraint.shape)

    print('soft - hard')
    print(soft_grad[0, ...] - hard_grad[0, ...])
    print('prob - hard')
    print(prob_constraint[0, ...] - hard_grad[0, ...])
