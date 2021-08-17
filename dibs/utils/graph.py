import functools 

import igraph as ig
import itertools
import os
import pathlib
import tqdm

import numpy as onp 

import jax.numpy as jnp
from jax import jit, vmap
from jax.ops import index, index_update
from jax import random

from dibs.exceptions import InvalidCPDAGError


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


def random_consistent_expansion(*, key, cpdag):
    """
    Generates a "consistent extension" DAG of a CPDAG as defined by 
    https://www.jmlr.org/papers/volume3/chickering02b/chickering02b.pdf
    i.e. a graph where DAG and CPDAG have the same skeleton and v-structures
    and every directed edge in the CPDAG has the same direction in the DAG

    This is achieved using the algorithm of
    http://ftp.cs.ucla.edu/pub/stat_ser/r185-dor-tarsi.pdf

    Every DAG in the MEC is a consistent extension of the corresponding CPDAG.

    Args:
        key: rng
        cpdag: adjacency matrix of a CPDAG[n_vars, n_vars] 
            breaks if it is not a valid CPDAG (merely a PDAG)
            (i.e. if cannot be extended to a DAG, e.g. undirected ring graph)

    Returns:
        [n_vars, n_vars] : adjacency matrix of a DAG consistent with the CPDAG
    """

    # check whether there are any undirected edges at all
    if jnp.sum(cpdag == cpdag.T) == 0:
        return cpdag

    G = cpdag.copy()
    A = cpdag.copy()

    N = A.shape[0]
    n_left = A.shape[0]
    node_exists = jnp.ones(A.shape[0])

    key, subk = random.split(key)
    ordering = random.permutation(subk, N)

    while n_left > 0:

        # find i satisfying:
        #   1) no directed edge leaving i (i.e. sink)
        #   2) undirected edge (i, j) must have j adjacent to all adjacent nodes of i
        #      (to avoid forming new v-structures when directing j->i)
        # If a valid CPDAG is input, then such an i must always exist, as every DAG in the MEC of a CPDAG is a consistent extension

        found_any_valid_candidate = False
        for i in ordering:

            if node_exists[i] == 0:
                continue
            
            # no outgoing _directed_ edges: (i,j) doesn't exist, or, (j,i) also does
            directed_i_out = A[i, :] == 1
            directed_i_in = A[:, i] == 1

            is_sink = jnp.all((1 - directed_i_out) | directed_i_in)
            if not is_sink: 
                continue 

            # for each undirected neighbor j of sink i
            i_valid_candidate = True
            undirected_neighbors_i = (directed_i_in == 1) & (directed_i_out == 1)
            for j in jnp.where(undirected_neighbors_i)[0]:

                # check that adjacents of i are a subset of adjacents j
                # i.e., check that there is no adjacent of i (ingoring j) that is not adjacent to j
                adjacents_j = (A[j, :] == 1) | (A[:, j] == 1)
                is_not_j = jnp.arange(N) != j
                if jnp.any(directed_i_in & (1 - adjacents_j) & is_not_j):
                    i_valid_candidate = False 
                    break 

            # i is valid, orient all edges towards i in consistent extension
            # and delete i and all adjacent egdes
            if i_valid_candidate:

                found_any_valid_candidate = True

                # to orient G towards i, delete (oppositely directed) i,j edges from adjacency
                G = index_update(G, index[i, jnp.where(undirected_neighbors_i)], 0)

                # remove i in A
                A = index_update(A, index[i, :], 0)
                A = index_update(A, index[:, i], 0)
                
                node_exists = index_update(node_exists, index[i], 0)

                n_left -= 1

                break
        
        if not found_any_valid_candidate:
            err_msg = (
                'found_any_valid_candidate = False; unable to create random consistent extension of CPDAG: ' + adjmat_to_str(cpdag) +
                ' | G: ' + adjmat_to_str(G) +
                ' | A: ' + adjmat_to_str(A) + 
                ' | ordering : ' + str(ordering.tolist())
            )
            raise InvalidCPDAGError(err_msg)
        
    return G


def make_all_dags(n_vars, try_to_load=True, try_to_save=True, return_matrices=False):
    """
    Generates all DAGs with N vertices by simply checking all possible adjacency matrices.
    Returns list of adjacency matrices.
    """

    if n_vars >= 6:
        raise ValueError('Superexponential, dont call with N >= 6')

    # try to load
    mod_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    main_dir = mod_dir.parent.parent
    fname = 'all_dags'
    path = main_dir / 'store' / f'{fname}_{n_vars}.npy'
    if try_to_load:
        if os.path.isfile(path):
            # saved as igraph.Graph objects
            dags = jnp.load(path, allow_pickle=True)
            if return_matrices:
                mats = []
                for g_ in dags:
                    mats.append(graph_to_mat(g_))
                return jnp.array(mats)
            else:
                return dags

    dags = []
    n_possible_edges = n_vars * n_vars

    # simply check all possible adjacency matrices
    for i, flat_mat in enumerate(tqdm.tqdm(itertools.product([0, 1], repeat=n_possible_edges), total=2 ** n_possible_edges)):
        mat = onp.array(flat_mat).reshape(n_vars, n_vars)
        g_ = mat_to_graph(mat)
        if g_.is_dag():
            dags.append(g_)

    if try_to_save:
        jnp.save(path, dags, allow_pickle=True)

    if return_matrices:
        mats = []
        for g_ in dags:
            mats.append(graph_to_mat(g_))
        return jnp.array(mats)
    else:
        return dags



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

def get_mat_idx(*, mat, mats, axis=(-1, -2)):
    """
    Returns first index of `mat` [...] in `mats` [n, ...]
    """
    idx = jnp.all(mats - mat == 0, axis=axis)
    return idx.argmax()

def get_undirected_edges(mat):
    return jnp.where((mat == mat.T) & (mat == 1))

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
