import functools 

import igraph as ig
import jax.numpy as jnp
from jax import jit, vmap



@functools.partial(jit, static_argnums=(1,))
def acyclic_constr(mat, n_vars):
    """
        Differentiable acyclicity constraint from
        Yu et al 2019
        http://proceedings.mlr.press/v97/yu19a/yu19a.pdf

        mat:  [n_vars, n_vars]
        out:  [1, ], [n_vars, n_vars]   constraint value and gradient w.r.t. mat

    """

    alpha = 1.0 / n_vars
    # M = jnp.eye(n_vars) + alpha * mat * mat # [original version]
    M = jnp.eye(n_vars) + alpha * mat

    # one less power, to have gradient readily available
    M_mult = jnp.linalg.matrix_power(M, n_vars - 1)

    # h = (M_mult.T * M).sum() - n_vars # bit faster, but correctness less obvious
    h = jnp.trace(M_mult @ M) - n_vars

    # gradient_h = M_mult.T * mat * 2 # [original version]
    gradient_h = M_mult.T 

    return h, gradient_h

@functools.partial(jit, static_argnums=(1,))
def acyclic_constr_nograd(mat, n_vars):
    """
        Differentiable acyclicity constraint from
        Yu et al 2019
        http://proceedings.mlr.press/v97/yu19a/yu19a.pdf

        mat:  [n_vars, n_vars]
        out:  [1, ] constraint value 

    """

    alpha = 1.0 / n_vars
    # M = jnp.eye(n_vars) + alpha * mat * mat # [original version]
    M = jnp.eye(n_vars) + alpha * mat

    M_mult = jnp.linalg.matrix_power(M, n_vars)
    h = jnp.trace(M_mult) - n_vars
    return h

elwise_acyclic_constr_nograd = jit(vmap(acyclic_constr_nograd, (0, None), 0), static_argnums=(1,))

@functools.partial(jit, static_argnums=(1,))
def eltwise_acyclic_constr(mat, n_vars):
    """
        Elementwise (batched) differentiable acyclicity constraint from  
        Yu et al 2019
        http://proceedings.mlr.press/v97/yu19a/yu19a.pdf

        mat:  [N, n_vars, n_vars]
        out:  [N, ], [N, n_vars, n_vars]   constraint value and gradient w.r.t. mat

    """

    alpha = 1.0 / n_vars
    # M = jnp.eye(n_vars) + alpha * mat * mat  # [original version]
    M = jnp.eye(n_vars) + alpha * mat

    # one less power, to have gradient readily available
    M_mult = jnp.linalg.matrix_power(M, n_vars - 1)

    #h = jnp.trace(jnp.einsum('nab,nbc->nac', M_mult, M), axis1=-2, axis2=-1) - n_vars
    h = jnp.trace(M_mult @ M, axis1=-2, axis2=-1) - n_vars

    # gradient_h = M_mult.transpose((0, 2, 1)) * mat * 2 # [original version]
    gradient_h = M_mult.transpose((0, 2, 1))

    return h, gradient_h


def graph_to_mat(G):
    """Returns adjacency matrix of ig.Graph object """
    return jnp.array(G.get_adjacency().data)

def mat_to_graph(W):
    """Returns ig.Graph object for adjacency matrix """
    return ig.Graph.Weighted_Adjacency(W.tolist())

def mat_is_dag(W):
    """Returns True iff adjacency matrix represents a DAG"""
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def adjmat_to_str(mat, max_len=40):
    """
    Converts {0,1}-adjacency matrix to human-readable string
    """

    edges_mat = jnp.where(mat == 1)
    undir_ignore = set() # undirected edges, already printed

    def get_edges():
        for e in zip(*edges_mat):
            u, v = e
            # undirected?
            if mat[v, u] == 1:
                # check not printed yet
                if e not in undir_ignore:
                    undir_ignore.add((v, u))
                    yield (u, v, True) 
            else:
                yield (u, v, False)

    strg = '  '.join([(f'{e[0]}--{e[1]}' if e[2] else
                       f'{e[0]}->{e[1]}') for e in get_edges()])
    if len(strg) > max_len:
        return strg[:max_len] + ' ... '
    elif strg == '':
        return '<empty graph>'
    else:
        return strg



if __name__ == '__main__':

    for N in range(1, 5):
        dags = make_all_dags(N)
        print(N, len(dags))
