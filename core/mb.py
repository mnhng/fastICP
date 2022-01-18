import numpy as np
from sklearn.linear_model import lasso_path


def get_MB_lasso(X, Y, size):
    _, coefs, _ = lasso_path(X, Y, n_alphas=5*size, tol=0.01, max_iter=10000)
    cz = np.apply_along_axis(np.sum, 0, coefs != 0)

    def tmp(m):
        if np.sum(cz == m):
            return np.where(cz == m)
        else:
            return tmp(m - 1)

    mm = tmp(size)[0][0]
    return np.where(coefs[:, mm] != 0)[0].tolist()
