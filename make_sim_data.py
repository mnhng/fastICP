#!/usr/bin/env python3
import argparse
import json
import pickle
import pathlib

import numpy as np

import sim


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pred', '-p', type=int, required=True, help='number of predictors (i.e., variables different from E and Y)')
    parser.add_argument('--n_int', '-i', type=int, required=True, help='samples the number of interventions in each graph')
    parser.add_argument('--d_type', '-t', required=True, help='data type')
    parser.add_argument('--h_int_str', type=float, help='intervention strength')
    parser.add_argument('--s_int_str', type=float, help='soft intervention reduction strength')
    parser.add_argument('--n_int_str', type=float, help='noise intervention magnitude')
    parser.add_argument('--m_str', type=float, help='mirror noise')
    parser.add_argument('--p_connection', '-c', type=float, help='probability of any single edge being present in the graph')
    parser.add_argument('--b_g', '-g', type=int, required=True, help='number of graphs to sample')
    parser.add_argument('--b_d', '-d', type=int, required=True, help='number of datasets to sample per graph')
    parser.add_argument('--nseq', '-n', type=int, nargs='+', required=True,  help='sequence of sample sizes to try')
    parser.add_argument('--out_dir', '-o', type=pathlib.Path, required=True)

    return parser.parse_args()


def main(args):
    p_connection = args.p_connection if args.p_connection else 2 / args.n_pred
    assert not args.out_dir.exists(), 'trying to overwrite existing data'

    for rep in range(args.b_g):
        root = args.out_dir/f'rep_{rep}'

        root.mkdir(parents=True, exist_ok=True)

        n_int = 1 + np.random.choice(-args.n_int) if args.n_int < 0 else args.n_int
        graph = sim.SampleDAG(args.n_pred, n_int, p_connection)

        with open(root/'graph.txt', 'w') as fh:
            print(graph.nodes, graph.edges, sep='\n', file=fh)

        if args.d_type == 'gmirror':
            data_obj = sim.DataLinGaussianHardMirror(graph, args.h_int_str, args.m_str)
        elif args.d_type == 'ghard':
            data_obj = sim.DataLinGaussianHard(graph, args.h_int_str)
        elif args.d_type == 'gsoft':
            data_obj = sim.DataLinGaussianSoft(graph, args.s_int_str)
        elif args.d_type == 'gnoise':
            data_obj = sim.DataLinGaussianNoise(graph, args.n_int_str)
        elif args.d_type == 'ng':
            data_obj = sim.DataLinNGHard(graph, args.h_int_str)
        elif args.d_type == 'sq':
            data_obj = sim.DataSqGaussianHard(graph, args.h_int_str)
        elif args.d_type == 'mix':
            data_obj = sim.DataMixGaussianHard(graph, args.h_int_str)
        elif args.d_type == 'mult':
            data_obj = sim.DataMultGaussianHard(graph, args.h_int_str)
        else:
            raise ValueError(args.d_type)

        conf = {k: v for k, v in vars(args).items() if k != 'out_dir'}
        conf['data_obj_path'] = 'data_gen.bin'
        with open(root/conf['data_obj_path'], 'wb') as fh:
            pickle.dump(data_obj, fh)

        conf['data_paths'] = []
        (root/'data').mkdir(parents=True, exist_ok=True)
        i = 0
        for _ in range(args.b_d):
            data_obj.BuildCoefMatrix()
            for n in args.nseq:
                record = {'N': int(n), 'path': f'data/{i}.npz'}
                E, X, Y = data_obj.MakeData(n)
                np.savez(root/record['path'], E=E, X=X, Y=Y)
                conf['data_paths'].append(record)
                i += 1

        with open(root/'config.json', 'w') as fh:
            json.dump(conf, fh, indent=4)


if __name__ == '__main__':
    main(get_args())
