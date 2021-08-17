import os
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision

import collections
import pickle

import jax.numpy as jnp
from jax.ops import index, index_add, index_update

import rpy2.robjects as robjects

STORE_ROOT_DATA = ['data']

if __name__ == '__main__':

    '''
    Loads R datasets and converts them to Python adjacency and coefficient matrices

    Data from: 
        https://www.bnlearn.com/bnrepository/
        https://www.bnlearn.com/documentation/man/bn.fit.class.html

    '''

    suffix = '.rda'
    data_filenames = [
        'gaussian-bn/ecoli70',
        'gaussian-bn/magic-irri',
        'gaussian-bn/magic-niab',
    ]

    for data_filename in data_filenames:

        # load into R
        path = os.path.abspath(os.path.join('..', *STORE_ROOT_DATA, data_filename + suffix))
        robjects.r('load("' + path + '")')


        # get node names
        robjects.r('node_names <- attributes(bn)$names')
        node_names = list(robjects.r['node_names'])
        n_nodes = len(node_names)
        assert(len(node_names) == len(set(node_names)))
        node_names_dict = {node: id for id, node in enumerate(node_names)}

        incidence_list = dict()
        coefflist = dict()

        # extract strings
        n_edges = 0
        for node in node_names:

            # get parents
            robjects.r(f'parents_{node} <- bn${node}$parents')
            parents = list(robjects.r[f'parents_{node}'])
            incidence_list[node] = parents
            n_edges += len(parents)

            # get coefficients
            coeffs = dict()
            robjects.r(f'coeff_{node}_intercept000 <- bn${node}$coefficients["(Intercept)"]')
            coeffs['intercept000'] = robjects.r[f'coeff_{node}_intercept000'][0]

            for par in parents:
                robjects.r(f'coeff_{node}_{par} <- bn${node}$coefficients["{par}"]')
                coeffs[par] = robjects.r[f'coeff_{node}_{par}'][0]

            coefflist[node] = coeffs

        # convert to matrices
        adjmat = jnp.zeros((n_nodes, n_nodes), dtype=jnp.int32)

        coeffmat = jnp.zeros((n_nodes, n_nodes), dtype=jnp.float32)
        biasmat = jnp.zeros((n_nodes,), dtype=jnp.float32)

        for node in node_names:

            node_idx = node_names_dict[node]

            # weights
            for par in incidence_list[node]:

                par_idx = node_names_dict[par]
                weight = coefflist[node][par]

                assert(adjmat[par_idx, node_idx] == 0)
                assert(coeffmat[par_idx, node_idx] == 0.0)

                adjmat =   index_update(adjmat,   index[par_idx, node_idx], 1)
                coeffmat = index_update(coeffmat, index[par_idx, node_idx], weight)

            # biases
            bias = coefflist[node]['intercept000']
            assert(biasmat[node_idx] == 0.0)

            biasmat = index_update(biasmat, index[node_idx], bias)

        data = {
            'node_names_dict':    node_names_dict,
            'incidence_list':     incidence_list,
            'adjacency_matrix':   adjmat,
            'coefficient_list':   coefflist,
            'coefficient_matrix': coeffmat,
            'bias_vector':        biasmat,
        }

        print()
        print(data_filename + suffix)
        print(f'nodes: {n_nodes}  edges: {n_edges}')

        save_path = os.path.abspath(os.path.join('..', *STORE_ROOT_DATA, data_filename + '.pk'))
        with open(save_path, 'wb') as fp:
            pickle.dump(data, fp)

        print(f'Saved at:   {save_path}')

