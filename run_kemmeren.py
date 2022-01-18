#!/usr/bin/env python3
import argparse
import pathlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import core


#### Loading of data ####
observation, intervention, intervention_pos = [
    pd.read_csv(f'data/kemmeren/Kemmeren{spec}.csv.gz')
    for spec in ['Obs', 'Int', 'IntPos']
]
observation = observation[observation.columns[1:]].to_numpy()
intervention = intervention[intervention.columns[1:]].to_numpy()
intervention_pos = intervention_pos[intervention_pos.columns[1:]].to_numpy().squeeze()
# Fix indentation starting at 1
intervention_pos = (intervention_pos - 1).astype(int)

n_obs, _ = observation.shape
n_int, n_gene = intervention.shape


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', '-m', required=True)
    parser.add_argument('--target', '-t', type=int, help='target gene')
    parser.add_argument('--out', '-o', type=pathlib.Path, required=True)

    return parser.parse_args()


def generate_mask(n_obs_, n_int_, n_fold_, seed=0):
    rng = np.random.default_rng(seed)
    pool = np.arange(n_fold_).repeat(np.ceil(n_int_/n_fold_))

    obs_mask = np.ones(n_obs_, dtype=int) * (-1)
    int_mask = rng.choice(pool, size=n_int_, replace=False)
    heldout_ints = {i: np.where(int_mask == i)[0] for i in range(n_fold_)}

    return np.concatenate([int_mask, obs_mask]), heldout_ints


def get_idx(gene_x):
    """
    Return the experiment id of the intervention on gene_x. E.g. if 3rd intervention is on gene number 3687, get_idx(3687) -> 3
    """
    if gene_x is None:
        return None
    else:
        positions = np.where(gene_x == intervention_pos)[0]
        return positions[0] if len(positions) > 0 else None


#### Function defitions ####
def is_true_positive(gene_x, gene_y, threshold=0.01):
    """
    Function to check whether gene_x is a true cause of gene_y, in the sense that
    when intervening on gene_x, gene_y lies in the upper or lower 1% tail of its
    observational distribution.
    """
    lo, hi = np.quantile(observation[:, gene_y], q=[threshold, 1 - threshold])
    intervention_effect = intervention[get_idx(gene_x), gene_y]
    return (intervention_effect <= lo) | (intervention_effect >= hi)


def get_PA(gene_y):
    alpha = 0.01
    # 2. split data into folds
    n_fold = 3
    mask, heldout_ints = generate_mask(n_obs, n_int, n_fold_=n_fold)

    X = np.concatenate([intervention, observation])
    E = np.repeat([1, 0], [n_int, n_obs])

    # 1. exclude intervention on Y
    y_int_idx = get_idx(gene_y)
    if y_int_idx is not None:
        # delete rows doesn't have any re-indexing problem
        RE, RX = np.delete(E, [y_int_idx], axis=0), np.delete(X, [y_int_idx], axis=0)
        sm = np.delete(mask, [y_int_idx], axis=0)
    else:
        RE, RX = E, X
        sm = mask

    linear_test = core.InvariantResidualRegTest(alpha, LinearRegression)
    # 3. use n-1 folds to infer PA and remaining fold to check
    results = []
    n_parents = sum(is_true_positive(gene_x, gene_y) for gene_x in range(n_gene) if get_idx(gene_x) is not None)
    for i in range(n_fold):
        data = RX[sm != i]
        in_genes = np.asarray([j for j in range(n_gene) if j != gene_y])

        if args.method == 'ICP':
            found = core.ICP(RE[sm != i], data[:, in_genes], data[:, gene_y], linear_test, use_MB=True)
        elif args.method == 'MMSE_ICP':
            found = core.MMSE_ICP(RE[sm != i], data[:, in_genes], data[:, gene_y], linear_test, use_MB=True)
        elif args.method == 'fastICP':
            found = core.fastICP(RE[sm != i], data[:, in_genes], data[:, gene_y], linear_test, use_MB=True)
        else:
            raise ValueError(args.method)

        for j, pval_gene in found.items():
            gene_x = in_genes[j]
            x_int_idx = get_idx(gene_x)
            checkable = x_int_idx is not None and x_int_idx in heldout_ints[i]
            # print(j, gene_x, x_int_idx, checkable)
            if checkable:
                results.append({
                    'gene_y': gene_y,
                    'gene_x': gene_x,
                    'p_val': pval_gene,
                    'TP': is_true_positive(gene_x, gene_y),
                    'nPA': n_parents,
                })

    # insert dummy row if found nothing
    if not results:
        results.append({'gene_y': gene_y, 'gene_x': -1, 'p_val': 1, 'TP': False, 'nPA': n_parents, })

    return results


if __name__ == '__main__':
    args = get_args()

    if args.target < n_gene:
        frame = pd.DataFrame(get_PA(args.target))
        args.out.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.out/f'{args.target:04}.csv', index=False)
