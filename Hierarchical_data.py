import pandas as pd
import networkx as nx


from timeit import timeit

df = pd.DataFrame(
    {
        'child':     [3102, 2010, 3011, 3000, 3033, 2110, 3111, 2100],
        'parent':    [2010, 1000, 2010, 2110, 2100, 1000, 2110, 1000]
    },  columns=['child', 'parent']
)

def all_descendants_nx():
    DiG = nx.from_pandas_edgelist(df,'parent','child',create_using=nx.DiGraph())
    return pd.DataFrame.from_records([(n1,n2) for n1 in DiG.nodes() for n2 in nx.ancestors(DiG, n1)], columns=['descendant','ancestor'])

print(timeit(all_descendants_nx, number=50))

a = all_descendants_nx()

print(a)
