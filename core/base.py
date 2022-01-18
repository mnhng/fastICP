import itertools

import numpy as np
import scipy.linalg as linalg
import scipy.stats as stats
from sklearn.model_selection import cross_val_predict


def powerset(items, exclude_emptyset=True):
    out = []
    for sl in itertools.product(*[[[], [i]] for i in items]):
        out.append({j for i in sl for j in i})

    out = sorted(out, key=lambda x: len(x))

    if exclude_emptyset:
        return out[1:]

    return out


# AX = Y + eps
def linear_regress(X, Y):
    A = linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    return Y - X.dot(A)


def ttestAndLeveneCheck(E, resid):
    E_labels = np.unique(E)
    pvalue = 1
    for label in E_labels:
        in_group, out_group = resid[E == label], resid[E != label]
        invariant_mean = stats.ttest_ind(in_group, out_group).pvalue
        invariant_std = stats.levene(in_group, out_group).pvalue
        pvalue = min(pvalue, 2*min(invariant_mean, invariant_std))

    bonferroni_adj = 1 if len(E_labels) == 2 else len(E_labels)

    return bonferroni_adj * pvalue


class InvariantResidualCVRegTest():
    def __init__(self, alpha, model_cls, **param):
        self.alpha = alpha
        self.param = param
        self.model_cls = model_cls

    def check(self, E, X, Y, covar, **kwargs):
        assert isinstance(covar, (list, tuple)), type(covar)
        if len(covar):
            resid = Y - cross_val_predict(
                self.model_cls(**self.param),
                X[:, covar], Y, cv=10 if len(Y) < 500 else 2,
            )
        else:
            resid = Y

        pvalue = ttestAndLeveneCheck(E, resid)
        return {'pvalue': pvalue, 'is_invariant': pvalue >= self.alpha, 'mse': (resid**2).mean()}


class InvariantResidualRegTest():
    def __init__(self, alpha, model_cls, **param):
        self.alpha = alpha
        self.model_cls = model_cls
        self.param = param

    def check(self, E, X, Y, covar, **kwargs):
        assert isinstance(covar, (list, tuple)), type(covar)
        if len(covar):
            regr = self.model_cls(**self.param)
            inp = X[:, covar].reshape(-1, len(covar))
            regr.fit(inp, Y)
            resid = Y - regr.predict(inp)
        else:
            resid = Y

        pvalue = ttestAndLeveneCheck(E, resid)
        return {'pvalue': pvalue, 'is_invariant': pvalue >= self.alpha, 'mse': (resid**2).mean()}


def is_invariant(E, X, Y, covar):
    if covar is None or covar == []:
        resid = Y
    else:
        resid = linear_regress(X[:, covar].reshape(-1, len(covar)), Y)

    return ttestAndLeveneCheck(E, resid), resid**2
