import numpy as np
import networkx as nx


class BaseGen():
    def __init__(self, graph):
        self.g = graph
        self.lut = {name: i for i, name in enumerate(graph.nodes())}

    def BuildCoefMatrix(self):
        raise NotImplementedError()

    def _f_obs(self, X, node, **kwargs):
        raise NotImplementedError()

    def _f_int(self, E, X, node, **kwargs):
        raise NotImplementedError()

    def MakeData(self, N):
        X = np.empty((N, self.g.order()))
        E = X[:, self.lut['E']] = np.random.choice([0, 1], N)  # equal-size observation/intervention splits

        # assigning nodes one by one, following causal order
        for n in nx.topological_sort(self.g):
            if n == 'E':
                continue

            idx = self.lut[n]
            if self.g.has_edge('E', n):  # Nodes which are children of E
                X[:, idx] = self._f_int(E, X, n)
            else:  # Remaining nodes
                X[:, idx] = self._f_obs(X, n)
        X /= np.std(X, axis=0, keepdims=True)

        return E, np.delete(X, [self.lut['E'], self.lut['Y']], axis=1), X[:, self.lut['Y']]


class DataLinGaussian(BaseGen):
    def __init__(self, graph):
        super().__init__(graph)
        self.sd = 1

    def BuildCoefMatrix(self):
        d = self.g.order()
        self.coef_mat = np.random.uniform(0.5, 2, (d, d)) * np.random.choice([-1, 1], (d, d))

    def _f_obs(self, X, node, W=None, noise_std=None):
        # NOTE: require to output zero vector when there is no parent
        PA_idx = [self.lut[name] for name in self.g.predecessors(node) if name != 'E']

        if W is None:
            W = self.coef_mat
        weights = W[PA_idx, self.lut[node]].reshape(1, -1)

        f_PA = (weights * X[:, PA_idx]).sum(axis=1)
        eps = np.random.normal(0, self.sd if noise_std is None else noise_std, len(X))

        return f_PA + eps


class DataLinGaussianHard(DataLinGaussian):
    def __init__(self, graph, InterventionStrength):
        super().__init__(graph)
        self.InterventionStrength = InterventionStrength

    def _f_int(self, E, X, node):
        return (E==0) * self._f_obs(X, node) + (E==1) * self.InterventionStrength


class DataLinGaussianHardMirror(DataLinGaussianHard):
    def __init__(self, graph, InterventionStrength, mirror_noise):
        super().__init__(graph,  InterventionStrength)
        self.mirror_noise = mirror_noise

    def MakeData(self, N):
        E, X, Y = super().MakeData(N)
        X_copy = X + np.random.randn(*X.shape) * self.mirror_noise

        return E, np.concatenate([X, X_copy], axis=1), Y


class DataLinGaussianSoft(DataLinGaussian):
    def __init__(self, graph, ReducedStrength):
        super().__init__(graph)
        self.InterventionStrength = 1
        self.ReducedStrength = ReducedStrength

    def BuildCoefMatrix(self):
        super().BuildCoefMatrix()
        d = self.g.order()
        self.W_mod = np.random.uniform(.0, self.ReducedStrength, (d, d)) * self.coef_mat

    def _f_int(self, E, X, node):
        a1 = self._f_obs(X, node)
        if not [name for name in self.g.predecessors(node) if name != 'E']:
            # NOTE: If there is no parent other than E, fall back to hard intervention
            # because soft intervention yields identical result to no intervention.
            return (E==0) * a1 + (E==1) * self.InterventionStrength

        return (E==0) * a1 + (E==1) * self._f_obs(X, node, W=self.W_mod)


class DataLinGaussianNoise(DataLinGaussian):
    def __init__(self, graph, multiplier):
        super().__init__(graph)
        self.new_sd = self.sd * multiplier

    def _f_int(self, E, X, node):
        a1 = self._f_obs(X, node)
        a2 = self._f_obs(X, node, noise_std=self.new_sd)

        return (E==0) * a1 + (E==1) * a2


class DataLinNG(BaseGen):
    def __init__(self, graph):
        super().__init__(graph)
        self.sd = 1

    def BuildCoefMatrix(self):
        d = self.g.order()
        self.coef_mat = np.random.uniform(0.5, 2, (d, d)) * np.random.choice([-1, 1], (d, d))

    def _eps(self, N, std):
        K = np.random.uniform(.1, .5, N)
        M = np.random.normal(0, std, N)
        alpha = np.random.uniform(2, 4, N)
        return K * np.sign(M) * np.abs(M)**alpha

    def _f_obs(self, X, node, W=None, noise_std=None):
        # NOTE: require to output zero vector when there is no parent
        PA_idx = [self.lut[name] for name in self.g.predecessors(node) if name != 'E']

        if W is None:
            W = self.coef_mat
        weights = W[PA_idx, self.lut[node]].reshape(1, -1)

        f_PA = (weights * X[:, PA_idx]).sum(axis=1)
        eps = self._eps(len(X), self.sd if noise_std is None else noise_std)

        return f_PA + eps


class DataLinNGHard(DataLinNG):
    def __init__(self, graph, InterventionStrength):
        super().__init__(graph)
        self.InterventionStrength = InterventionStrength

    def _f_int(self, E, X, node):
        return (E==0) * self._f_obs(X, node) + (E==1) * self.InterventionStrength


class DataSqGaussian(BaseGen):
    def __init__(self, graph):
        super().__init__(graph)
        self.sd = 1

    def BuildCoefMatrix(self):
        d = self.g.order()
        self.coef_mat = np.random.uniform(0.5, 2, (d, d)) * np.random.choice([-1, 1], (d, d))

    def _f_obs(self, X, node, W=None, noise_std=None):
        # NOTE: require to output zero vector when there is no parent
        PA_idx = [self.lut[name] for name in self.g.predecessors(node) if name != 'E']

        if W is None:
            W = self.coef_mat
        weights = W[PA_idx, self.lut[node]].reshape(1, -1)

        f_PA = (weights * X[:, PA_idx]**2).sum(axis=1)
        eps = np.random.normal(0, self.sd if noise_std is None else noise_std, len(X))

        return f_PA + eps


class DataSqGaussianHard(DataSqGaussian):
    def __init__(self, graph, InterventionStrength):
        super().__init__(graph)
        self.InterventionStrength = InterventionStrength

    def _f_int(self, E, X, node):
        return (E==0) * self._f_obs(X, node) + (E==1) * self.InterventionStrength


class DataMixGaussian(BaseGen):
    def __init__(self, graph):
        super().__init__(graph)
        self.sd = 1

    def BuildCoefMatrix(self):
        d = self.g.order()
        self.coef_mat = np.random.uniform(0.5, 2, (d, d)) * np.random.choice([-1, 1], (d, d))
        self.func_mat = np.random.randint(0, 4, (d, d))

    def _f(self, blk, func_idxs):
        assert len(blk.shape) == 2 and blk.shape[1] == len(func_idxs)
        f1 = blk
        f2 = np.where(blk > 0, blk, 0)
        f3 = np.sign(blk)*np.sqrt(np.abs(blk))
        f4 = np.sin(2*np.pi*blk)

        slabs = [f1, f2, f3, f4]
        slabs = np.concatenate([np.expand_dims(mat, -1) for mat in slabs], axis=-1)

        return slabs[:, np.arange(len(func_idxs)), func_idxs]

    def _f_obs(self, X, node, W=None, noise_std=None):
        # NOTE: require to output zero vector when there is no parent
        node_idx = self.lut[node]
        PA_idx = [self.lut[name] for name in self.g.predecessors(node) if name != 'E']

        if W is None:
            W = self.coef_mat
        weights = W[PA_idx, node_idx].reshape(1, -1)
        func_idxs = self.func_mat[PA_idx, node_idx]

        f_PA = (weights * self._f(X[:, PA_idx], func_idxs)).sum(axis=1)
        eps = np.random.normal(0, self.sd if noise_std is None else noise_std, len(X))

        return f_PA + eps


class DataMixGaussianHard(DataMixGaussian):
    def __init__(self, graph, InterventionStrength):
        super().__init__(graph)
        self.InterventionStrength = InterventionStrength

    def _f_int(self, E, X, node):
        return (E==0) * self._f_obs(X, node) + (E==1) * self.InterventionStrength


class DataMultGaussian(BaseGen):
    def __init__(self, graph):
        super().__init__(graph)
        self.sd = 1

    def BuildCoefMatrix(self):
        d = self.g.order()
        self.coef_mat = np.random.choice([-1, 1], (d, d))
        self.func_mat = np.random.randint(0, 4, (d, d))

    def _f(self, blk, func_idxs):
        assert len(blk.shape) == 2 and blk.shape[1] == len(func_idxs)
        f1 = blk
        f2 = np.where(blk > 0, blk, 0)
        f3 = np.sign(blk)*np.sqrt(np.abs(blk))
        f4 = np.sin(2*np.pi*blk)

        slabs = [f1, f2, f3, f4]
        slabs = np.concatenate([np.expand_dims(mat, -1) for mat in slabs], axis=-1)

        return slabs[:, np.arange(len(func_idxs)), func_idxs]

    def _f_obs(self, X, node, W=None, noise_std=None):
        # NOTE: require to output zero vector when there is no parent
        node_idx = self.lut[node]
        PA_idx = [self.lut[name] for name in self.g.predecessors(node) if name != 'E']

        if W is None:
            W = self.coef_mat
        weights = W[PA_idx, node_idx].reshape(1, -1)
        func_idxs = self.func_mat[PA_idx, node_idx]

        f_PA = (weights * self._f(X[:, PA_idx], func_idxs)).prod(axis=1)
        eps = np.random.normal(0, self.sd if noise_std is None else noise_std, len(X))

        return f_PA + eps


class DataMultGaussianHard(DataMultGaussian):
    def __init__(self, graph, InterventionStrength):
        super().__init__(graph)
        self.InterventionStrength = InterventionStrength

    def _f_int(self, E, X, node):
        return (E==0) * self._f_obs(X, node) + (E==1) * self.InterventionStrength
