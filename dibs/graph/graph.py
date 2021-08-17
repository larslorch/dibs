import numpy as np
import tqdm
import scipy
import igraph as ig
import random as pyrandom

import jax.numpy as jnp
from jax import random
from jax.ops import index, index_mul

from dibs.utils.graph import mat_to_graph, graph_to_mat, mat_is_dag

class GraphDistribution:
    """
    Class to represent distributions over graphs.
    """

    def __init__(self, n_vars, verbose=False):
        self.n_vars = n_vars
        self.verbose = verbose

    def sample_G(self, return_mat=False):
        raise NotImplementedError

    def log_normalization_constant(self, *, all_g):
        """
        Computes normalization constant for log p(G), i.e. `Z = log(sum_G p(g))`

        Args:
            all_g: list of igraph.Graph objects

        Returns:
            float
        """
        log_prob_g_unn = np.zeros(len(all_g))
        for i, g in enumerate(tqdm.tqdm(all_g, desc='p(G) log_normalization_constant', disable=not self.verbose)):
            log_prob_g_unn[i] = self.unnormalized_log_prob(g=g)
        log_prob_sum_g = scipy.special.logsumexp(log_prob_g_unn)
        return log_prob_sum_g

    def unnormalized_log_prob_single(self, *, g, j):
        """
        p(G) ~ 1

        Args:
            g: igraph.Graph
            j: node index
        
        Returns:
            float
        """
        return 0.0

    def unnormalized_log_prob(self, *, g):
        """
        p(G) ~ 1

        Args:
            g: igraph.Graph
        
        Returns:
            float
        """
        return 0.0


    def unnormalized_log_prob_soft(self, *, soft_g):
        """
        p(G) ~ 1

        Args:
            soft_g: [d, d] soft adjacency matrix with values in [0,1]
        
        Returns:
            float

        """
        return 0.0


class ErdosReniDAGDistribution(GraphDistribution):
    """
    Randomly oriented Erdos-Reni random graph 
    with prior p(G) = const
    """

    def __init__(self, n_vars, verbose=False, n_edges=None):
        super(ErdosReniDAGDistribution, self).__init__(n_vars=n_vars, verbose=verbose)

        self.n_vars = n_vars
        self.n_edges = n_edges or 2 * n_vars
        self.p = self.n_edges / ((self.n_vars * (self.n_vars - 1)) / 2)

        self.verbose = verbose

    def sample_G(self, key, return_mat=False):
        """Samples DAG"""

        key, subk = random.split(key)
        mat = random.bernoulli(subk, p=self.p, shape=(self.n_vars, self.n_vars)).astype(jnp.int32)

        # make DAG by zeroing above diagonal; k=-1 indicates that diagonal is zero too
        dag = jnp.tril(mat, k=-1)

        # randomly permute
        key, subk = random.split(key)
        P = random.permutation(subk, jnp.eye(self.n_vars, dtype=jnp.int32))
        dag_perm = P.T @ dag @ P

        if return_mat:
            return dag_perm
        else:
            g = mat_to_graph(dag_perm)
            return g

    def unnormalized_log_prob_single(self, *, g, j):
        """
        p(G) ~ p^|E| (1-p)^((n choose 2) - |E|)
        """
        parent_edges = g.incident(j, mode='in')
        n_parents = len(parent_edges)
        return n_parents * jnp.log(self.p) + (self.n_vars - n_parents - 1) * jnp.log(1 - self.p)

    def unnormalized_log_prob(self, *, g):
        """
        p(G) ~ p^|E| (1-p)^((n choose 2) - |E|)
        """
        N = self.n_vars * (self.n_vars - 1) / 2.0
        E = len(g.es)

        return E * jnp.log(self.p) + (N - E) * jnp.log(1 - self.p)

    def unnormalized_log_prob_soft(self, *, soft_g):
        """
        p(G) ~ p^|E| (1-p)^((n choose 2) - |E|)
        """
        N = self.n_vars * (self.n_vars - 1) / 2.0
        E = soft_g.sum()
        return E * jnp.log(self.p) + (N - E) * jnp.log(1 - self.p)


class ScaleFreeDAGDistribution(GraphDistribution):
    """
    Randomly oriented Scale-free random graph 
    with prior p(G) = const
    """

    def __init__(self, n_vars, verbose=False, n_edges_per_node=2):
        super(ScaleFreeDAGDistribution, self).__init__(
            n_vars=n_vars, verbose=verbose)

        self.n_vars = n_vars
        self.n_edges_per_node = n_edges_per_node
        self.verbose = verbose

    def sample_G(self, key, return_mat=False):
        """Samples DAG"""

        pyrandom.seed(key.sum())
        perm = random.permutation(key, self.n_vars).tolist()
        g = ig.Graph.Barabasi(n=self.n_vars, m=self.n_edges_per_node, directed=True).permute_vertices(perm)

        if return_mat:
            return graph_to_mat(g)
        else:
            return g

    def unnormalized_log_prob_single(self, *, g, j):
        """
        p(G) ~ prod_j deg(j)^-3
        """
        parent_edges = g.incident(j, mode='in')
        n_parents = len(parent_edges)
        return -3 * jnp.log(1 + n_parents)

    def unnormalized_log_prob(self, *, g):
        """
        p(G) ~ prod_j deg(j)^-3
        """
        return jnp.array([self.unnormalized_log_prob_single(g=g, j=j) for j in range(self.n_vars)]).sum()

    def unnormalized_log_prob_soft(self, *, soft_g):
        """
        p(G) ~ prod_j deg(j)^-3
        """
        soft_indegree = soft_g.sum(0)
        return jnp.sum(-3 * jnp.log(1 + soft_indegree))


class LowerTriangularDAGDistribution(GraphDistribution):
    """
    A distribution over DAGs sampling a lower-triangular matrix
    and permuting it randomly.
    This is not uniformly random, because e.g. the empty graph
    occurs n! more often than the chain 1 -> 2 -> ... -> N
    However, it can be used to encode the belief about the number of edges
    """

    def __init__(self, n_vars, verbose=False, sparsity_factor=None):
        super(LowerTriangularDAGDistribution, self).__init__(n_vars=n_vars, verbose=verbose)
        self.n_vars = n_vars
        self.verbose = verbose

        if sparsity_factor is None:
            self.p = 0.5
        else:
            # if p > (1 + epsilon) ln(n) / n, the graph is connected almost surely as n -> infty
            self.p = sparsity_factor * jnp.log(self.n_vars) / self.n_vars

    def sample_G(self, key, return_mat=False):
        """Samples DAG
            This sampling procedure is merely used to generate DAGs with a certain
            distribution of the number of edges and is by no means uniformly random.
        """

        key, subk = random.split(key)
        mat = random.bernoulli(subk, p=self.p, shape=(self.n_vars, self.n_vars)).astype(jnp.int32)

        # make DAG by zeroing above diagonal; k=-1 indicates that diagonal is zero too
        dag = np.tril(mat, k=-1)

        # randomly permute
        key, subk = random.split(key)
        P = random.permutation(subk, jnp.eye(self.n_vars, dtype=jnp.int32))
        dag_perm = P.T @ dag @ P

        if return_mat:
            return dag_perm
        else:
            g = mat_to_graph(dag_perm)
            return g

    def unnormalized_log_prob_single(self, *, g, j):
        """
        p(G) ~ p^|E| (1-p)^((n choose 2) - |E|)
        """
        parent_edges = g.incident(j, mode='in')
        n_parents = len(parent_edges)
        return n_parents * jnp.log(self.p) + (self.n_vars - n_parents - 1) * jnp.log(1 - self.p)


    def unnormalized_log_prob(self, *, g):
        """
        p(G) ~ p^|E| (1-p)^((n choose 2) - |E|)
        """
        N = self.n_vars * (self.n_vars - 1) / 2.0
        E = len(g.es)

        return E * jnp.log(self.p) + (N - E) * jnp.log(1 - self.p)

    def unnormalized_log_prob_soft(self, *, soft_g):
        """
        p(G) ~ p^|E| (1-p)^((n choose 2) - |E|)
        """
        N = self.n_vars * (self.n_vars - 1) / 2.0
        E = soft_g.sum()

        return E * jnp.log(self.p) + (N - E) * jnp.log(1 - self.p)



class UniformDAGDistributionRejection(GraphDistribution):
    """
    Uniform distribution over DAGs
    """

    def __init__(self, n_vars, verbose=False):
        super(UniformDAGDistributionRejection, self).__init__(n_vars=n_vars, verbose=verbose)
        self.n_vars = n_vars 
        self.verbose = verbose

    def sample_G(self, key, return_mat=False):
        """Samples uniformly random DAG by rejection sampling
            Prohibitively inefficient for n > 5
        """
        mask_idx = index[..., jnp.arange(self.n_vars), jnp.arange(self.n_vars)]

        while True:
            key, subk = random.split(key)
            mat = random.bernoulli(subk, p=0.5, shape=(self.n_vars, self.n_vars)).astype(jnp.int32)
            mat = index_mul(mat, mask_idx, 0)

            if mat_is_dag(mat):
                if return_mat:
                    return mat
                else:
                    return mat_to_graph(mat)
