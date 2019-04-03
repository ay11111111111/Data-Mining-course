import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy

#%%
tt = pd.read_csv('bp-data.txt', header=None)
G = nx.Graph(tt)
A = nx.adjacency_matrix(G).todense()
plt.imshow(A, interpolation='none')
plt.show()
nx.draw(G)
plt.show()
print(len(G), nx.is_connected(G))

#%%
def hierarchy_clustering(G):
    N = len(G)
    S = np.zeros((N, N))

    floyd = nx.floyd_warshall(G)
    for i in range(N):
        for j in range(N):
            S[i, j] = floyd[i][j]
    # print(S)

    Y = squareform(S)
    Z = hierarchy.linkage(Y, method='average')
    hierarchy.dendrogram(Z)
    plt.show()
    clusters = hierarchy.fcluster(Z, 3.25, criterion='distance')
    n_clusters = max(clusters)
    print(n_clusters)

    nodes = np.arange(N)
    for i in range(1, n_clusters + 1):
        GG = G.subgraph(nodes[clusters == i])
        # print(len(GG), GG.edges())
        nx.draw(GG)
        plt.show()

    return clusters

print(hierarchy_clustering(G))
print(nx.is_connected(G))
