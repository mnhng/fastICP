#!/usr/bin/env python3
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_gene_exp_rank(data_dict):
    x = np.arange(cutoff) + 1

    fig, axe = plt.subplots(figsize=(4, 4))

    axe.plot([0, cutoff], [0, cutoff], linestyle='dotted', color='k')

    dx = 0
    for name, data in data_dict.items():
        axe.plot(x + dx/10, data, label=name)
        dx += 1

    axe.legend()
    axe.set_xticks(np.linspace(0, cutoff, 11, dtype=int))
    axe.set_yticks(np.linspace(0, cutoff, 11, dtype=int))
    axe.set_xlabel('Number of Predictions')
    axe.set_ylabel('Number of True Positives')

    fig.tight_layout()
    return fig


if __name__ == '__main__':
    root = pathlib.Path('kemmeren')
    RESULTS = {}
    RESULTS['ICP'] = pd.concat(pd.read_csv(fp) for fp in (root/'ICP').glob('*.csv')).sort_values('p_val', ascending=True)
    RESULTS['MMSE_ICP'] = pd.concat(pd.read_csv(fp) for fp in (root/'MMSE_ICP').glob('*.csv')).sort_values('p_val', ascending=True)
    RESULTS['fastICP'] = pd.concat(pd.read_csv(fp) for fp in (root/'fastICP').glob('*.csv')).sort_values('p_val', ascending=True)
    cutoff = 40
    data_dict = {k: p['TP'].to_numpy() for k, p in RESULTS.items()}
    data_dict = {k: s[:cutoff].cumsum() for k, s in data_dict.items()}

    plot_gene_exp_rank(data_dict).savefig(f'kemmeren.png', dpi=300)
