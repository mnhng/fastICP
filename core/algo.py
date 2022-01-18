from .base import powerset
from .func import *
from .mb import get_MB_lasso

__all__ = ['ICP', 'MMSE_ICP', 'fastICP']


def ICP(E, X, Y, test_obj, use_MB=False):
    test_result = test_obj.check(E, X, Y, [])
    if test_result['is_invariant']:
        return dict()

    test_sets = powerset(get_MB_lasso(X, Y, 10) if use_MB else range(X.shape[1]))

    invariant_sets, test_results = [], [(set(), test_result['pvalue'])]
    S_star, best = set(), float('inf')
    for x in test_sets:
        if any([s <= x for s in invariant_sets]):
            continue

        test_result = test_obj.check(E, X, Y, list(x))
        test_results.append((x, test_result['pvalue']))
        if test_result['is_invariant']:
            invariant_sets.append(x)

    S_star = set.intersection(*invariant_sets) if len(invariant_sets) else set()
    return est_var_pval(S_star, test_results)


def MMSE_ICP(E, X, Y, test_obj, use_MB=False):
    test_result = test_obj.check(E, X, Y, [])
    if test_result['is_invariant']:
        return dict()

    test_sets = powerset(get_MB_lasso(X, Y, 10) if use_MB else range(X.shape[1]))

    invariant_sets, test_results = [], [(set(), test_result['pvalue'])]
    S_star, best = set(), float('inf')
    for x in test_sets:
        if any([s <= x for s in invariant_sets]):
            continue

        test_result = test_obj.check(E, X, Y, list(x))
        test_results.append((x, test_result['pvalue']))
        if test_result['is_invariant']:
            invariant_sets.append(x)
            if best > test_result['mse']:
                S_star, best  = x, test_result['mse']

    return est_var_pval(S_star, test_results)


def fastICP(E, X, Y, test_obj, use_MB=False, verbose=False):
    test_result = test_obj.check(E, X, Y, [])
    if test_result['is_invariant']:
        return dict()

    test_results = [(set(), test_result['pvalue'])]
    S_star = get_MB_lasso(X, Y, 100) if use_MB else list(range(X.shape[1]))

    S_star = find_invariant(E, X, Y, test_obj, S_star, test_results)
    if S_star is None:
        return dict()  # TODO: should return None instead (no invariant test found)

    S_star = quad_pruning(E, X, Y, test_obj, S_star, test_results)

    return est_var_pval(S_star, test_results)
