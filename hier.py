import numpy as np
import pandas as pd  # only used to return a dataframe


def list_ancestors(edges):
    """
    Take edge list of a rooted tree as a numpy array with shape (E, 2),
    child nodes in edges[:, 0], parent nodes in edges[:, 1]
    Return pandas dataframe of all descendant/ancestor node pairs

    Ex:
        df = pd.DataFrame({'child': [200, 201, 300, 301, 302, 400],
                           'parent': [100, 100, 200, 200, 201, 300]})

        df
           child  parent
        0    200     100
        1    201     100
        2    300     200
        3    301     200
        4    302     201
        5    400     300

        list_ancestors(df.values)

        returns

            descendant  ancestor
        0          200       100
        1          201       100
        2          300       200
        3          300       100
        4          301       200
        5          301       100
        6          302       201
        7          302       100
        8          400       300
        9          400       200
        10         400       100
    """
    ancestors = []
    for ar in trace_nodes(edges):
        ancestors.append(np.c_[np.repeat(ar[:, 0], ar.shape[1]-1),
                               ar[:, 1:].flatten()])
    return pd.DataFrame(np.concatenate(ancestors),
                        columns=['descendant', 'ancestor'])


def trace_nodes(edges):
    """
    Take edge list of a rooted tree as a numpy array with shape (E, 2),
    child nodes in edges[:, 0], parent nodes in edges[:, 1]
    Yield numpy array with cross-section of tree and associated
    ancestor nodes

    Ex:
        df = pd.DataFrame({'child': [200, 201, 300, 301, 302, 400],
                           'parent': [100, 100, 200, 200, 201, 300]})

        df
           child  parent
        0    200     100
        1    201     100
        2    300     200
        3    301     200
        4    302     201
        5    400     300

        trace_nodes(df.values)

        yields

        array([[200, 100],
               [201, 100]])

        array([[300, 200, 100],
               [301, 200, 100],
               [302, 201, 100]])

        array([[400, 300, 200, 100]])
    """
    mask = np.in1d(edges[:, 1], edges[:, 0])
    gen_branches = edges[~mask]
    edges = edges[mask]
    yield gen_branches
    while edges.size != 0:
        mask = np.in1d(edges[:, 1], edges[:, 0])
        next_gen = edges[~mask]
        gen_branches = numpy_col_inner_many_to_one_join(next_gen, gen_branches)
        edges = edges[mask]
        yield gen_branches


def numpy_col_inner_many_to_one_join(ar1, ar2):
    """
    Take two 2-d numpy arrays ar1 and ar2,
    with no duplicate values in first column of ar2
    Return inner join of ar1 and ar2 on
    last column of ar1, first column of ar2

    Ex:

        ar1 = np.array([[1,  2,  3],
                        [4,  5,  3],
                        [6,  7,  8],
                        [9, 10, 11]])

        ar2 = np.array([[ 1,  2],
                        [ 3,  4],
                        [ 5,  6],
                        [ 7,  8],
                        [ 9, 10],
                        [11, 12]])

        numpy_col_inner_many_to_one_join(ar1, ar2)

        returns

        array([[ 1,  2,  3,  4],
               [ 4,  5,  3,  4],
               [ 9, 10, 11, 12]])
    """
    ar1 = ar1[np.in1d(ar1[:, -1], ar2[:, 0])]
    ar2 = ar2[np.in1d(ar2[:, 0], ar1[:, -1])]
    if 'int' in ar1.dtype.name and ar1[:, -1].min() >= 0:
        bins = np.bincount(ar1[:, -1])
        counts = bins[bins.nonzero()[0]]
    else:
        counts = np.unique(ar1[:, -1], False, False, True)[1]
    left = ar1[ar1[:, -1].argsort()]
    right = ar2[ar2[:, 0].argsort()]
    return np.concatenate([left[:, :-1],
                           right[np.repeat(np.arange(right.shape[0]),
                                           counts)]], 1)