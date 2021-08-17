import random
import numpy as np
import igraph as ig


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def simulate_dag(n_vars, n_edges, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        n_vars (int): num of nodes
        n_edges (int): expected num of edges
        graph_type (str): erdos, scale-free, bipartite

    Returns:
        B (np.ndarray): [n_vars, n_vars] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'erdos':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=n_vars, m=n_edges)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)

    elif graph_type == 'scale-free':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=n_vars, m=int(round(n_edges / n_vars)), directed=True)
        B = _graph_to_adjmat(G)

    elif graph_type == 'bipartite':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * n_vars)
        G = ig.Graph.Random_Bipartite(
            top, n_vars - top, m=n_edges, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)

    else:
        raise ValueError('unknown graph type')
    
    B_perm = _random_permutation(B)
    # B_perm = B

    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def simulate_parameters(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """Simulate SEM parameters for a DAG.

    Args:
        B (np.ndarray): [n_vars, n_vars] binary adj matrix of DAG
        w_ranges (tuple): weight ranges; each edge has equal probability 
            of being drawn uniformly from a given range

    Returns:
        W (np.ndarray): [n_vars, n_vars] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W


def simulate_linear_sem(W, n_samples, noise_scale=1.0, noise_type='uniform'):
    """Compute linear SEM for a weighted adjacency matrix.

    Args:
        W (np.ndarray): [n_vars, n_vars] weighted adj matrix of DAG
        n_samples (int): number of samples from SEM
        noise_type (str): noise type

    Returns:
        X (np.ndarray): [n_samples, n_vars] `n_samples` observations of all variables according to SEM
    """
    n_vars = W.shape[0]
    assert(is_dag(W))

    # sample noise
    if noise_type == 'uniform':
        Z = noise_scale * np.random.uniform(-1, 1, size=(n_samples, n_vars))
    else:
        raise ValueError('invalid noise_type')

    # get topological ordering
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    top_order = G.topological_sorting()

    # forward-sample BN in topological order
    X = np.zeros_like(Z)
    for j in top_order:
        assert(np.allclose(X[:, j], 0)) # current variables should not have been set yet
        X[:, j] = X @ W[:, j] + Z[:, j]

    return X

if __name__ == '__main__':

    n_samples, n_vars, n_edges, graph, noise_scale = 100, 5, 10, 'scale-free', 1.0
    
    # binary adjacency marix of DAG
    B = simulate_dag(n_vars, n_edges, graph)

    # real weighted adjacency matrix as linear SEM parameters
    W = simulate_parameters(B)
    
    # compute SEM
    X = simulate_linear_sem(W, n_samples, noise_scale=noise_scale)
   
