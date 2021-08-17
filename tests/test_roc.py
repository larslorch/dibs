import os
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision

import collections
import pprint
import tqdm
import time
import ast
import numpy as onp
import pandas as pd

import jax.numpy as jnp
from jax import grad, jit, vmap, vjp, jvp, jacfwd, jacrev
from jax import random
import jax.lax as lax
from jax.scipy.special import logsumexp

import matplotlib.pyplot as plt

from sklearn import metrics

from dibs.utils.func import bit2id, id2bit

STORE_ROOT = ['results']

if __name__ == '__main__':

    jnp.set_printoptions(precision=4, suppress=True)
    key = random.PRNGKey(0)

    descr = 'roc_test.csv'
    path = os.path.abspath(os.path.join(
        '..', *STORE_ROOT, descr
    ))

    g_gt = jnp.array([
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ])
    g_gt_flat = g_gt.reshape(-1)


    g_pred1 = jnp.array([
        [0.1, 0.5, 0.3, 0.9],
        [0.0, 0.6, 0.2, 0.2],
        [0.2, 0.2, 0.0, 0.1],
        [0.2, 0.2, 0.0, 0.1],
    ])
    g_pred2 = jnp.array([
        [0.1, 0.3, 0.1, 0.5],
        [0.0, 0.4, 0.9, 0.7],
        [0.0, 0.1, 0.0, 0.4],
        [0.0, 0.1, 0.0, 0.4],
    ])

    # test df
    a = [
        ['method 1', g_pred1.reshape(-1).tolist()],
        ['method 2', g_pred2.reshape(-1).tolist()],
    ]
    orig_df = pd.DataFrame(a, columns=['descr', 'roc'])
    print(orig_df)

    # to csv
    orig_df.to_csv(path)
    print(path)
    print()

    # from csv
    to_jnp_array = lambda lst: jnp.array(ast.literal_eval(lst), dtype=jnp.float64)
    df = pd.read_csv(path, index_col=0)
    df['roc'] = df['roc'].apply(to_jnp_array)

    # roc curve
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    for i, row in df.iterrows():
        print(row['descr'])

        # roc
        fpr, tpr, _ = metrics.roc_curve(g_gt_flat, row['roc'])
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr)
        display.plot(ax=axs[0], label=row['descr'])

        print('AUC', roc_auc)

        # precision-recall
        precision, recall, thresholds = metrics.precision_recall_curve(g_gt_flat, row['roc'])
        display = metrics.PrecisionRecallDisplay(precision=precision, recall=recall)
        display.plot(ax=axs[1], label=row['descr'])

    plt.show()


  
