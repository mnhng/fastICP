import numpy as np
import networkx as nx


def SampleDAG(d, N_Interventions, p_conn):
    """
    Samples a dag with d + 2 nodes (d predictors (X), E and Y) by first sampling the
    d + 1 sized subgraph of (X, Y) where each node is connected with probability p_conn.
    Then, we sample N_Interventions edges from a new node E to nodes in the graph (X, Y)
    and these edges and the root node E are added. Then, we sample the position of Y among
    all nodes that 1) are not children of E and 2) are descendants of E.
        d: Number of predictor nodes
        N_interventions: Number of nodes to intervene on
        p_conn: Probability of connecting to edges in the graph of (X, Y)
    """
    assert (N_Interventions >= 1) and (N_Interventions <= d), 'N_interventions must be in {1, ..., d}'

    PossibleY = set()
    try_max = 50000
    D1 = d + 1
    for _ in range(try_max):
        A = np.random.choice([0, 1], (D1, D1), p=[1 - p_conn, p_conn])
        A[np.tril_indices(D1)] = 0  # NOTE: this ensure that range(1, d) is topological order
        IntPos = np.random.choice([i + 1 for i in range(D1)], N_Interventions, replace=False)
        Interventions = np.array([[1 if i in IntPos else 0 for i in range(D1 + 1)]])
        A_ = np.append(Interventions,
                       np.append(np.zeros((D1, 1)), A, axis=1),
                       axis=0)
        g = nx.convert_matrix.from_numpy_array(A_, create_using=nx.DiGraph)
        PossibleY = nx.descendants(g, 0) - set(g.successors(0))
        if len(PossibleY) > 0:
            break

    if len(PossibleY) == 0:
        print(f'Graph could not be sampled in {try_max} attempts.')
        return

    Ynode = np.random.choice(list(PossibleY))

    return nx.relabel_nodes(g, {0: 'E', Ynode: 'Y'} | dict(zip([x for x in range(1, D1+1) if x != Ynode], range(d))))
