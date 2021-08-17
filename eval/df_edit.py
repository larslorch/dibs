import os
import argparse
import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax import random
from collections import defaultdict


def _make_edit_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs='+', required=True, help="path to dfs")
    parser.add_argument("--first_name_replace", nargs='+', required=True, help="name replacement of <first arg> with <second arg>")

    # option
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--remove_by_column_val", nargs='+')

    return parser


if __name__ == '__main__':

    parser = _make_edit_parser()
    kwargs = parser.parse_args()

    # single file
    if len(kwargs.paths) == 1:
        print('Single file processing')
        assert(len(kwargs.first_name_replace) == 2)
        assert(kwargs.first_name_replace[0] != kwargs.first_name_replace[1])

        path = kwargs.paths[0]
        assert(kwargs.first_name_replace[0] in path)
        df = pd.read_csv(path, index_col=0)
        print('old', path)
        print('--', len(df))

        # options
        if kwargs.remove_by_column_val:
            args = kwargs.remove_by_column_val
            assert(len(args) >= 2 and len(args) % 2 == 0)
            for j in range(len(args) // 2):
                df = df[df[args[2*j]] != args[2*j+1]]
                print('rem', args[2*j], args[2*j+1])
                print('--', len(df))


        # assemble new file
        new_path = os.path.join(os.path.split(path)[0], os.path.split(path)[1].replace(*kwargs.first_name_replace))
        print('new', new_path)
        df.to_csv(new_path)
            

    # multi file
    else:
        print('Multi file processing')
        assert(len(kwargs.first_name_replace) == 2)
        assert(kwargs.first_name_replace[0] in kwargs.paths[0])

        dfs = []
        for path in kwargs.paths:
            dfs.append(pd.read_csv(path, index_col=0))
            print('old', path)
            print('[', len(dfs[-1]), ']')

        # options
        if kwargs.merge:

            df = pd.concat(dfs, ignore_index=True)  # .reset_index()
            old_path = kwargs.paths[0]
            new_path = os.path.join(os.path.split(old_path)[0], os.path.split(old_path)[1].replace(*kwargs.first_name_replace))
            print('new', new_path)
            df.to_csv(new_path)
            print('--', len(df))

        else:
            raise ValueError('No valid options passed')
