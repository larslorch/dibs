import os
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision

import numpy as onp # needed for np.unique(axis=0)

import jax.numpy as jnp
import jax.lax as lax
from jax import jit, vmap
from jax.ops import index, index_add, index_update
from jax.lax import stop_gradient
from jax.scipy.special import logsumexp
from jax.tree_util import tree_flatten, tree_unflatten, tree_map, tree_multimap, tree_reduce
from jax import random


def expand_by(arr, n):
    """
    Expands jnp.array by n dimensions at the end
    """
    return jnp.expand_dims(arr, axis=tuple(arr.ndim + j for j in range(n)))


@jit
def sel(mat, mask):
    '''
        jit/vmap helper function

        mat:   [N, d]
        mask:  [d, ]   boolean 

        returns [N, d] with columns of `mat` with `mask` == 1 non-zero a
        and the columns with `mask` == 0 are zero

        e.g. 
        mat 
        1 2 3
        4 5 6
        7 8 9

        mask
        1 0 1

        out
        1 0 3
        4 0 6
        7 0 9
    '''
    return jnp.where(mask, mat, 0)

@jit
def leftsel(mat, mask, maskval=0.0):
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
    valid_indices = jnp.where(
        mask, jnp.arange(mask.shape[0]), mask.shape[0])
    padded_mat = jnp.concatenate(
        [mat, maskval * jnp.ones((mat.shape[0], 1))], axis=1)
    padded_valid_mat = padded_mat[:, jnp.sort(valid_indices)]
    return padded_valid_mat


def mask_topk_(x, topkk):
    '''
        x       [N, ]
        topk:   int
        returns indices of `topk` entries of `x` in decreasing order
    '''
    mask = x.argsort()[-topkk:][::-1]
    return mask

mask_topk = jit(mask_topk_, static_argnums=(1,))

@jit
def bit2id(b):
    '''
        in:   [N, d, d] with {0,1} values
        out:  [N, ] with integer bit representation
    '''
    N, d, _ = b.shape
    b_flat = b.reshape(N, d * d)
    return jnp.packbits(b_flat, axis=1, bitorder='little')


def id2bit_(id, d):
    '''
        in:   
            id: [N, ?] with integer bit representation and number of vars rep. by of matrix
            d:  number of variables
            
        out:  [N, d, d] with {0,1} values
    '''
    N, _ = id.shape
    b_flat = jnp.unpackbits(id, axis=1, bitorder='little')
    b_flat = b_flat[:, :d * d]
    return b_flat.reshape(N, d, d)

id2bit = jit(id2bit_, static_argnums=(1,))

@jit
def kullback_leibler_vec(*, p, q=None, log_q=None):
    """
    Computes empirical K(p || q)
    as: 
        sum_i p[i] log (p[i] / q[i])

    If `log_q` is provided, ignores q for stability.
    """
    zero_mask = jnp.zeros_like(p)
    if log_q is None:
        if q is None:
            raise ValueError('Provide either q or log q')
        
        # stable and jit-table log q
        log_q = lax.select(
            q > 0.0,
            jnp.log(q),
            jnp.inf * zero_mask
        )
    
    # jit-table entropy of p
    return jnp.sum(p * (lax.select(p > 0.0, jnp.log(p), jnp.zeros_like(p)) - log_q))

# not jittable because output shape depends on number of unique elements
# haven't managed to re-write jnp.unique 
def particle_empirical(b):
    '''
        b:      [N, d, d] with {0,1} values
        out:    tuple where tuple[0] contains unique ids 
                            tuple[1] contains the empirical log probability

        not jittable because of jnp.unique
    '''
    N, _, _ = b.shape
    ids = bit2id(b)
    unique, counts = onp.unique(ids, axis=0, return_counts=True)

    # empirical using counts
    log_probs = jnp.log(counts) - jnp.log(N)

    return unique, log_probs


def particle_empirical_mixture(b, eltwise_log_prob):
    '''
        b:      [N, d, d] with {0,1} values
        eltwise_log_prob:   [N, d, d] -> [N,]
        out:    tuple where tuple[0] contains unique ids 
                            tuple[1] contains the empirical log probability

        not jittable because of jnp.unique
    '''
    n_vars = b.shape[-1]
    ids = bit2id(b)
    unique, counts = onp.unique(ids, axis=0, return_counts=True)

    # mixture using relative log probs
    log_probs = eltwise_log_prob(id2bit(unique, n_vars))
    log_probs -= logsumexp(log_probs)

    return unique, log_probs


def particle_joint_empirical(b, theta):
    '''
        b:      [N, d, d] with {0,1} values
        theta:  [N, ...] 
        out:    tuple where tuple[0] contains ids 
                            tuple[1] contains thetas
                            tuple[2] contains the empirical log probability

        not jittable because of jnp.unique
    '''
    N, _, _ = b.shape
    ids = bit2id(b)

    # empirical
    log_probs = - jnp.log(N) * jnp.ones(N)

    return ids, theta, log_probs


def particle_joint_mixture(b, theta, eltwise_log_prob):
    '''
        b:      [N, d, d] with {0,1} values
        theta:  [N, ...] 
        eltwise_log_prob:   [N, d, d], [N, ...] -> [N,]

        out:    tuple where tuple[0] contains ids 
                            tuple[1] contains thetas
                            tuple[2] contains the empirical log probability

        not jittable because of jnp.unique
    '''
    N, n_vars, _ = b.shape
    ids = bit2id(b)

    # mixture using relative log probs
    # assumes that every particle is unique (always true because of theta)
    log_probs = eltwise_log_prob(id2bit(ids, n_vars), theta)
    log_probs -= logsumexp(log_probs)

    return ids, theta, log_probs


def dist_is_none(dist):
    '''
    Checks whether distribution tuple `dist` is None
    '''
    return (dist[0] is None) or (dist[1] is None)


@jit
def log_prob_id(id, dist):
    '''
        id:         [1, ]    unique id of a graph
        dist:       [N, 2]   log distribution representation

        returns     [1, ]    log prob of id in dist
    '''
    dist_ids, dist_log_probs = dist
    is_match = jnp.all(id == dist_ids, axis=1)
    return lax.cond(
        jnp.any(is_match),
        lambda _: dist_log_probs[is_match.argmax(0)],
        lambda _: -jnp.inf,
        operand=None)

log_prob_ids = jit(vmap(log_prob_id, (0, None), 0))


@jit
def log_prob_id_finite(id, dist):
    '''
        id:         [1, ]    unique id of a graph
        dist:       [N, 2]   log distribution representation

        returns     [1, ]    log prob of id in dist
    '''
    dist_ids, dist_log_probs = dist
    is_match = jnp.all(id == dist_ids, axis=1)
    return lax.cond(
        jnp.any(is_match),
        lambda _: dist_log_probs[is_match.argmax(0)],
        lambda _: -jnp.array(1000.0), # probability of 0 is set to 1e-1000 to have a non-inf KL
        operand=None)

log_prob_ids_finite = jit(vmap(log_prob_id_finite, (0, None), 0))


def kullback_leibler_dist(p, q, finite=False):
    """
    Computes empirical K(p || q)
    as: 
        sum_i p[i] log (p[i] / q[i])

    using 
        p  log distribution tuple
        q  log distribution tuple
    """
    ids, logpx = p
    px = jnp.exp(logpx)
    logqx = log_prob_ids(ids, q) if not finite else log_prob_ids_finite(ids, q)
    return jnp.sum(px * (logpx - logqx))


@jit
def pairwise_squared_norm(*, x, y, axis=None):
    """Computes pairwise squared euclidean norm
    where
        x:  [N, ...]
        y:  [M, ...]

    returns
        returns shape [N, M] where elt i,j is  ||x[i] - y[j]||_2^2
    """
    # this assert is not necessary but function only works for these two settings for convenience
    assert(((x.ndim == 1 or x.ndim == 2) and (y.ndim == 1 or y.ndim == 2)) or
           ((x.ndim == 3 or x.ndim == 4) and (y.ndim == 3 or y.ndim == 4)))

    xndim_was_singular = x.ndim == 1 or x.ndim == 3 # covers case where inputs are [N, d, k, 2]
    yndim_was_singular = y.ndim == 1 or y.ndim == 3
    if xndim_was_singular:
        x = jnp.expand_dims(x, axis=0)
    if yndim_was_singular:
        y = jnp.expand_dims(y, axis=0)

    assert(x.ndim == y.ndim and x.ndim >= 2)

    # all but first axis is usually used for the norm, assuming that first dim is batch dim
    if axis is None:
        axis = tuple(i for i in range(1, x.ndim))


    # pairwise sum of squares along `axis` [N x M]
    pointwise_sq = jnp.sum(x ** 2, axis=axis).reshape(-1, 1) \
                 + jnp.sum(y ** 2, axis=axis)

  
    # pairwise product, i.e. outer product [N x M]
    if len(axis) == 3:
        outer_product = jnp.einsum('Nijk,Mijk->NM', x, y)
    elif len(axis) == 2:
        outer_product = jnp.einsum('Nij,Mij->NM', x, y)
    else:
        outer_product = jnp.einsum('Ni,Mi->NM', x, y)

    # (x - y)^2 = x^2 + y^2 - 2 x^T y
    squared_norm = pointwise_sq - 2 * outer_product

    if xndim_was_singular:
        squared_norm = jnp.squeeze(squared_norm, axis=0)
    if yndim_was_singular: 
        squared_norm = jnp.squeeze(squared_norm, axis=0 if xndim_was_singular else 1)

    return squared_norm

def pairwise_squared_norm_mat(*, x, y):
    """Computes pairwise squared euclidean norm
    same as `pairwise_squared_norm` but works specifically for [d, d] matrices
    where
        x:  [N, d, d]
        y:  [M, d, d]

    returns
        returns shape [N, M] where elt i,j is  ||x[i] - y[j]||_2^2
    """
    assert((x.ndim == 2 or x.ndim == 3) and (y.ndim == 2 or y.ndim == 3))

    xndim_was_singular = x.ndim == 2
    yndim_was_singular = y.ndim == 2
    if xndim_was_singular:
        x = jnp.expand_dims(x, axis=0)
    if yndim_was_singular:
        y = jnp.expand_dims(y, axis=0)

    assert(x.ndim == y.ndim and x.ndim >= 2)

    # pairwise sum of squares along `axis` [N x M]
    pointwise_sq = jnp.sum(x ** 2, axis=(1, 2)).reshape(-1, 1) \
                 + jnp.sum(y ** 2, axis=(1, 2))

  
    # pairwise product, i.e. outer product [N x M]
    outer_product = jnp.einsum('Nij,Mij->NM', x, y)

    # (x - y)^2 = x^2 + y^2 - 2 x^T y
    squared_norm = pointwise_sq - 2 * outer_product

    if xndim_was_singular:
        squared_norm = jnp.squeeze(squared_norm, axis=0)
    if yndim_was_singular: 
        squared_norm = jnp.squeeze(squared_norm, axis=0 if xndim_was_singular else 1)

    return squared_norm

@jit
def squared_norm_pytree(x, y):

    """Computes squared euclidean norm between two pytrees
    where
        x:  PyTree 
        y:  PyTree 

    returns
        returns shape [] 
    """ 

    diff = tree_multimap(jnp.subtract, x, y)
    squared_norm_ind = tree_map(lambda leaf: jnp.square(leaf).sum(), diff)
    squared_norm = tree_reduce(jnp.add, squared_norm_ind)
    return squared_norm

pairwise_squared_norm_pytree_full = vmap(vmap(squared_norm_pytree, (None, 0), 0), (0, None), 0)

def pairwise_squared_norm_pytree_(x, y, singular_dim_theta):
    """Computes pairwise squared euclidean norm between two batched pytrees
    Assumes 
    where
        x:  PyTree with leading dimension [N, ...]
        y:  PyTree with leading dimension [M, ...]

    returns
        returns shape [N, M] where elt i,j is  ||x[i] - y[j]||_2^2
    """    

    # know whether or not we deal with a degenerate batch from first leaf
    # if so, expand dim to have consistent shapes in pairwise norm
    xndim_was_singular = tree_flatten(x)[0][0].ndim == singular_dim_theta
    yndim_was_singular = tree_flatten(y)[0][0].ndim == singular_dim_theta

    if xndim_was_singular:
        x = tree_map(lambda leaf: jnp.expand_dims(leaf, axis=0), x)
    if yndim_was_singular:
        y = tree_map(lambda leaf: jnp.expand_dims(leaf, axis=0), y)

    # squared norm
    squared_norm = pairwise_squared_norm_pytree_full(x, y)
    
    # undo shape changes
    if xndim_was_singular:
        squared_norm = jnp.squeeze(squared_norm, axis=0)
    if yndim_was_singular: 
        squared_norm = jnp.squeeze(squared_norm, axis=0 if xndim_was_singular else 1)

    return squared_norm

pairwise_squared_norm_pytree = jit(pairwise_squared_norm_pytree_, static_argnums=(2, ))


@jit
def pairwise_angular_distance(*, x, y):
    """Computes pairwise angular distance, where distance between 
    vectors u, v is defined as

        dist(u, v) = sum_over_all_dims arccos(u.v / |u|*|v|)

    which results in dist = `pi` if u = -v and and dist = 0 if u = v
    
        x:  [N, d, k, 2]
        y:  [M, d, k, 2]

    returns
        returns shape [N, M] where elt i,j is  dist(x[i], y[j])
    """

    # convert both to 4 dimensional tensor
    assert((x.ndim == 3 or x.ndim == 4) and 
            (y.ndim == 3 or y.ndim == 4))

    xndim_was_singular = x.ndim == 3
    yndim_was_singular = y.ndim == 3
    if xndim_was_singular:
        x = jnp.expand_dims(x, axis=0)
    if yndim_was_singular:
        y = jnp.expand_dims(y, axis=0)

    # unit vectors
    x = x / jnp.linalg.norm(x, axis=2, ord=None, keepdims=True)
    y = y / jnp.linalg.norm(y, axis=2, ord=None, keepdims=True)
    
    # inner products over k dimension
    inner_prods_u = jnp.einsum('Ndk,Mdk->NMd', x[:, :, :, 0], y[:, :, :, 0])
    inner_prods_v = jnp.einsum('Ndk,Mdk->NMd', x[:, :, :, 1], y[:, :, :, 1])

    # arccos(x) = nan for x > 1 and due to numerical instability this can occur
    # the below shrinks x (clip would cut gradients)
    inner_prods_u /= jnp.array(1.0 + 1e-5)
    inner_prods_v /= jnp.array(1.0 + 1e-5)

    # convert to distance via arccos
    dist = jnp.sum(jnp.arccos(inner_prods_u), axis=2) \
         + jnp.sum(jnp.arccos(inner_prods_v), axis=2)
    
    if xndim_was_singular:
        dist = jnp.squeeze(dist, axis=0)
    if yndim_was_singular:
        dist = jnp.squeeze(dist, axis=0 if xndim_was_singular else 1)

    return dist


def median_heuristic_from_squared_norm_(x, is_norm):
    """Computes median heuristic based on _squared_ distance as used by SVGD
    where
        x :  squared distances
        is_norm: if `is_norm` is `True`, assumes sqrt(squard distance), i.e. actual norm, is passed

    returns
        h :  scaling factor in SE kernel as:     exp( - 1/h d(x, x'))
                where h = med(squared dist) / log n
    """

    # assumes that all dimensions have the same length
    # assert left out because it breaks JAX tracing
    N_ = x.shape[0] if x.ndim > 0 else 1

    # med_square = jnp.array([jnp.median(x)]) # necessary for vmap down the road
    med_square = jnp.median(x) # necessary for vmap down the road
    if is_norm:
        med_square = med_square ** 2

    h = med_square / jnp.log(N_ + 1)

    return h

median_heuristic_from_squared_norm = jit(median_heuristic_from_squared_norm_, static_argnums=(1,))


# @jit
# def pairwise_structural_hamming_distance(*, x, y, axis=None, atol=1e-8):
#     """Computes pairwise Structural Hamming distance, i.e. 
#     the number of edge insertions, deletions or flips in order to transform one graph to another 
#     Note: 
#         - this means, edge reversals do not double count
#         - this means, getting an undirected edge wrong only counts 1

#     where
#         x:  [N, ...]
#         y:  [M, ...]

#     returns
#         returns shape [N, M] where elt i,j is  SHD(x[i], y[j]) = sum(x[i] != y[j])
#     """
#     # all but first axis is usually used for the norm, assuming that first dim is batch dim
#     assert(x.ndim == 3 and y.ndim == 3)

#     is_undirected_edge_x = jnp.isclose(x, 1, atol=atol) & jnp.isclose(x.transpose((0, 2, 1)), 1, atol=atol)
#     is_undirected_edge_y = jnp.isclose(y, 1, atol=atol) & jnp.isclose(y.transpose((0, 2, 1)), 1, atol=atol)

#      ### undirected edge scores
#     undirected_missing = ((jnp.expand_dims(is_undirected_edge_x, axis=1) & jnp.expand_dims(~is_undirected_edge_y, axis=0)) |
#                           (jnp.expand_dims(~is_undirected_edge_x, axis=1) & jnp.expand_dims(is_undirected_edge_y, axis=0)))
#     shd_undirected = jnp.triu(undirected_missing, k=1).sum(axis=(2, 3)) # only count once

#     ### directed edge scores
#     # via computing pairwise differences
#     pw_diff = jnp.expand_dims(x, axis=1) - jnp.expand_dims(y, axis=0)

#     # True iff x[i,j] != y[i,j]
#     is_diff = 1 - jnp.isclose(pw_diff, 0, atol=atol)

#     # True iff (x[i,j] - y[i,j]) + (x[j,i] - y[j,i]) = 0
#     flipped_if_diff = jnp.isclose(pw_diff + pw_diff.transpose((0, 1, 3, 2)), 0, atol=atol)

#     # determine which difference is a flip and which not
#     is_not_flip = is_diff & (1 - flipped_if_diff)
#     is_flip = is_diff & flipped_if_diff

#     # count 
#     n_add_del = (~undirected_missing & is_not_flip).sum(axis=(2, 3))
#     n_flip = jnp.triu(~undirected_missing & is_flip, k=1).sum(axis=(2, 3)) # only count once
#     shd_directed = n_add_del + n_flip

#     shd = shd_directed + shd_undirected
#     return shd

def pairwise_structural_hamming_distance(*, x, y, axis=None, atol=1e-8):
    """Simpler implementation taken from cdt.SHD

    Computes pairwise Structural Hamming distance, i.e.
    the number of edge insertions, deletions or flips in order to transform one graph to another
        - this means, edge reversals do not double count
        - this means, getting an undirected edge wrong only counts 1

    where
        x:  [N, ...]
        y:  [M, ...]

    returns
        returns shape [N, M] where elt i,j is  SHD(x[i], y[j]) = sum(x[i] != y[j])
    """

    # all but first axis is usually used for the norm, assuming that first dim is batch dim
    assert(x.ndim == 3 and y.ndim == 3)

    # via computing pairwise differences
    pw_diff = jnp.abs(jnp.expand_dims(x, axis=1) - jnp.expand_dims(y, axis=0))
    pw_diff = pw_diff + pw_diff.transpose((0, 1, 3, 2))

    # ignore double edges
    pw_diff = jnp.where(pw_diff > 1, 1, pw_diff)
    shd = jnp.sum(pw_diff, axis=(2, 3)) / 2 

    return shd



@jit
def symlog(x, const=500.0):
    """
    Log-like transformation for negative and positive values
    Equivalent to matplotlibs `symlog`

    https://iopscience.iop.org/article/10.1088/0957-0233/24/2/027001/meta
    """

    return jnp.sign(x) * jnp.log(1 + jnp.abs(x / const))
