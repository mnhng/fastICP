import numpy as np


def jaccard(A, B):
    # |A cap B| / |A cup B|, for sets A and B
    u = set.union(A, B)
    return len(set.intersection(A, B)) / len(u) if u else 1


def recall(pred, true):
    return len(set.intersection(pred, true)) / len(true) if true else int(len(pred) == 0)


def precision(pred, true):
    return len(set.intersection(pred, true)) / len(pred) if pred else int(len(pred) == len(true))


def f1(pred, true):
    p, r = precision(pred, true), recall(pred, true)
    if p + r == 0:
        return 0
    return 2*p*r/(p + r)


def to_adj_mat(node_list, adj_list=None, edge_list=None):
    D = len(node_list)
    val2idx = {val: idx for idx, val in enumerate(node_list)}
    out = np.zeros((D, D))
    assert adj_list is None or edge_list is None

    if edge_list is not None:
        for src, tgt in edge_list:
            out[val2idx[src], val2idx[tgt]] = 1

    if adj_list is not None:
        for src, tgts in adj_list.items():
            out[val2idx[src], [val2idx[i] for i in tgts]] = 1

    return out


def edge_errors(pred_adj_mat, true_adj_mat):
    """
    Counts all types of edge errors (false negatives, false positives, reversed edges)
    """
    diff = (true_adj_mat != 0).astype(int) - (pred_adj_mat != 0).astype(int)

    rev = (((diff + diff.transpose()) == 0) & (diff != 0)).sum() / 2
    # Each reversed edge necessarily leads to one fp and one fn so we need to subtract those
    fn = (diff == 1).sum() - rev
    fp = (diff == -1).sum() - rev

    return fn, fp, rev


def shd(pred_adj_mat, true_adj_mat):
    return sum(edge_errors(pred_adj_mat, true_adj_mat))


def score(pred, PA, AN, S_star):
    return {
            'JcPA': jaccard(pred, PA),
            'JcAN': jaccard(pred, AN),
            'JcS*': jaccard(pred, S_star),

            'Em': pred <= set(),

            'InPA': pred <= PA,
            'InAN': pred <= AN,
            'InS*': pred <= S_star,

            'EqS*': pred == S_star,
            'len': len(pred),

            'F1': f1(pred, PA),
            'Pr': precision(pred, PA),
            'Re': recall(pred, PA),

            'F1S*': f1(pred, S_star),
            'PrS*': precision(pred, S_star),
            'ReS*': recall(pred, S_star),

            'F1AN': f1(pred, AN),
            'PrAN': precision(pred, AN),
            'ReAN': recall(pred, AN),
    }
