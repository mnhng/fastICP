import networkx as nx


def true_MB(graph, target):
    PA_Y = set(graph.predecessors(target))
    CH_Y = set(graph.successors(target))
    PACH = list()
    for i in CH_Y:
        PACH.extend(list(graph.predecessors(i)))

    return set.union(PA_Y, CH_Y, set(PACH)) - {target}
