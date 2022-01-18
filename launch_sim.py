#!/usr/bin/env python3
import argparse
import json
import pathlib
import pickle
import time

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

import core
import metrics


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', '-i', type=pathlib.Path, nargs='+', required=True)
    parser.add_argument('--out', '-o', type=pathlib.Path, default='out')
    parser.add_argument('--method', '-m', required=True)
    parser.add_argument('--alpha', type=float, default=0.05)

    return parser.parse_args()


def run(folder, args):
    root = pathlib.Path(folder)

    with open(root/'data_gen.bin', 'rb') as fh:
        data_obj = pickle.load(fh)

    with open(root/'config.json') as fh:
        conf = json.load(fh)

    print(folder, data_obj.g.edges)

    AN_Y = nx.ancestors(data_obj.g, 'Y') - {'E'}
    PA_Y = set(data_obj.g.predecessors('Y'))
    S_star = set.intersection(nx.descendants(data_obj.g, 'E'), PA_Y)

    rows = []
    for run in conf['data_paths']:
        row = {'N': run['N'], 'i': folder, 'n_int': conf['n_int'], 'name': args.method}

        archive = np.load(root/run['path'])
        E, X, Y = archive['E'], archive['X'], archive['Y']
        d = X.shape[1]
        start = time.process_time()

        linear_test = core.InvariantResidualRegTest(args.alpha, LinearRegression)
        nl_cgb = core.InvariantResidualCVRegTest(args.alpha, GradientBoostingRegressor)

        if args.method == 'ICP':
            pred = {k for k in core.ICP(E, X, Y, linear_test, use_MB=True)}
        elif args.method == 'ICP_nl':
            pred = {k for k in core.ICP(E, X, Y, nl_cgb, use_MB=True)}
        elif args.method == 'MMSE_ICP':
            pred = {k for k in core.MMSE_ICP(E, X, Y, linear_test, use_MB=True)}
        elif args.method == 'MMSE_ICP_nl':
            pred = {k for k in core.MMSE_ICP(E, X, Y, nl_cgb, use_MB=True)}
        elif args.method == 'fastICP':
            pred = {k for k in core.fastICP(E, X, Y, linear_test)}
        elif args.method == 'fastICP_nl':
            pred = {k for k in core.fastICP(E, X, Y, nl_cgb)}
        else:
            raise ValueError(f'Unsupported method: {args.method}')

        row['time'] = time.process_time() - start

        rows.append(metrics.score(pred, PA=PA_Y, AN=AN_Y, S_star=S_star))

    return pd.DataFrame(rows)


def main(args):
    stem = f'{args.out}_{args.method}'
    full = pd.concat([run(d, args) for d in args.in_dir]).groupby(['N', 'i', 'name']).agg('mean')
    full.to_csv(f'{stem}.mean.csv')


if __name__ == '__main__':
    main(get_args())
