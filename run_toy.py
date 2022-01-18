#!/usr/bin/env python3
import numpy as np
import networkx as nx
from sklearn.linear_model import LinearRegression

import core
import sim


graph = nx.DiGraph({'E': [0], 0: [1, 2], 1: ['Y'], 2: ['Y']})

print(graph.nodes())
print(graph.edges())
node_names = [n for n in graph.nodes() if n not in {'E', 'Y'}]

data_obj = sim.DataLinGaussianHard(graph, InterventionStrength=1)
data_obj.BuildCoefMatrix()
np.set_printoptions(precision=3)
print(data_obj.coef_mat)

E, X, Y = data_obj.MakeData(1000)

linear_test = core.InvariantResidualRegTest(0.05, LinearRegression)

print('Parents of Y:', [node for node in graph.predecessors('Y') if node != 'E'])
print('Predicted parents of Y')
print('- ICP:', [node_names[i] for i in core.ICP(E, X, Y, linear_test)])
print('- MMSE_ICP:', [node_names[i] for i in core.MMSE_ICP(E, X, Y, linear_test)])
print('- fastICP:', [node_names[i] for i in core.fastICP(E, X, Y, linear_test)])
