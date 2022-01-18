import itertools

__all__ = ['find_invariant', 'quad_pruning', 'est_var_pval']


def est_var_pval(invariant_set, outcomes):
    out = {}
    for v in invariant_set:
        out[v] = max([pvalue for subset, pvalue in outcomes if v not in subset])

    return out


# NOTE: outcomes is a list that will be modified
def quad_pruning(E, X, Y, test_obj, starting_point, outcomes):
    out = starting_point
    while len(out):
        candidates = []
        for S in itertools.combinations(out, len(out)-1):
            new_result = test_obj.check(E, X, Y, S)
            outcomes.append((set(S), new_result['pvalue']))
            if new_result['is_invariant']:
                candidates.append((S, new_result['mse']))

        if not candidates:
            break
        else:
            out, _ = min(candidates, key=lambda x: x[1])

    return out


# NOTE: outcomes is a list that will be modified
def find_invariant(E, X, Y, test_obj, starting_point, outcomes, MAX_DEPTH=2):
    S_star = list(starting_point)

    candidates, depth = [], 1
    result = test_obj.check(E, X, Y, S_star)
    while not result['is_invariant'] and len(S_star):
        for S in itertools.combinations(S_star, len(S_star)-depth):
            new_result = test_obj.check(E, X, Y, S)
            outcomes.append((set(S), new_result['pvalue']))
            if new_result['pvalue'] > result['pvalue']:
                candidates.append((S, new_result))

        if len(candidates):
            S_star, result = max(candidates, key=lambda result: result[1]['pvalue'])
            candidates, depth = [], 1
        elif depth <= MAX_DEPTH:
            depth += 1
            if depth > len(S_star):
                return None
        else:
            return None

    return S_star
