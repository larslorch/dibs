import numpy as np
import igraph as ig


def make_toy_dags(d=4):
    """Returns toy DAGs"""
    
    Gs = []

    if d == 2:
        # 0
        g = ig.Graph(directed=True)
        g.add_vertices(d)
        g.add_edges([(0, 1)])
        Gs.append(g)

        # 1
        g = ig.Graph(directed=True)
        g.add_vertices(d)
        g.add_edges([(1, 0)])
        Gs.append(g)

    elif d == 3:
        # 0
        g = ig.Graph(directed=True)
        g.add_vertices(d)
        g.add_edges([(0, 1), (0, 2)])
        Gs.append(g)

        # 1
        g = ig.Graph(directed=True)
        g.add_vertices(d)
        g.add_edges([(0, 2), (1, 2)])
        Gs.append(g)

        # 2
        g = ig.Graph(directed=True)
        g.add_vertices(d)
        g.add_edges([(0, 1), (0, 2), (1, 2)])
        Gs.append(g)

    elif d == 4:
        # 0
        g = ig.Graph(directed=True)
        g.add_vertices(d)
        g.add_edges([(0, 1), (0, 2), (2, 3)])
        Gs.append(g)

        # 1
        g = ig.Graph(directed=True)
        g.add_vertices(d)
        g.add_edges([(0, 2), (1, 3), (2, 3)])
        Gs.append(g)

        # 2
        g = ig.Graph(directed=True)
        g.add_vertices(d)
        g.add_edges([(0, 1), (0, 2), (2, 1), (1, 3)])
        Gs.append(g)

        # 3
        g = ig.Graph(directed=True)
        g.add_vertices(d)
        g.add_edges([(0, 1), (0, 2), (2, 1), (1, 3), (2, 3)])
        Gs.append(g)

        # 4
        g = ig.Graph(directed=True)
        g.add_vertices(d)
        g.add_edges([(0, 1), (0, 2), (1, 3), (2, 3)])
        Gs.append(g)

    elif d == 5:
        # 0
        g = ig.Graph(directed=True)
        g.add_vertices(d)
        g.add_edges([(0, 1), (0, 2), (1, 2), (2, 3), (2, 4)])
        Gs.append(g)

        # 1
        g = ig.Graph(directed=True)
        g.add_vertices(d)
        g.add_edges([(0, 1), (1, 2), (0, 4), (4, 3), (3, 2)])
        Gs.append(g)

        # 2
        g = ig.Graph(directed=True)
        g.add_vertices(d)
        g.add_edges([(0, 1), (4, 0), (4, 1), (3, 4), (3, 0), (2, 1)])
        Gs.append(g)

        # 3
        g = ig.Graph(directed=True)
        g.add_vertices(d)
        g.add_edges([(1, 0), (4, 0), (0, 3), (0, 2)])
        Gs.append(g)

        # 4
        g = ig.Graph(directed=True)
        g.add_vertices(d)
        g.add_edges([(2, 4), (4, 0), (1, 0), (3, 4)])
        Gs.append(g)


    else:
        raise ValueError('Invalid toy graph size requested.')

    for g in Gs:
        assert(g.is_dag())
    return Gs

