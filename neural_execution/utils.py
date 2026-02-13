import random
import networkx as nx
"""

G = nx.erdos_renyi_graph(n=6, p=0.4)

print(f"노드: {list(G.nodes())}")
print(f"엣지: {list(G.edges())}")
print(f"연결됨: {nx.is_connected(G)}")
"""

def generate_graph(n_nodes, p=0.3):
    while True:
        G = nx.erdos_renyi_graph(n=n_nodes, p=p)

        if nx.is_connected(G):
            break

    for u, v in G.edges():
        G[u][v]['weight'] = random.randint(1, 10)

    return G

def bfs_trace(G, source):
    n = G.number_of_nodes()

    # 각 노드의 방문 여부 -> 0: 비방문, 1 -> 방문
    visited = [0] * n
    visited[source] = 1

    traces = []
    traces.append(list(visited))

    queue = [source]

    while queue:
        next_queue = []
        for node in queue:
            for neighbor in G.neighbors(node):
                if visited[neighbor] == 0:
                    visited[neighbor] = 1
                    next_queue.append(neighbor)

        if next_queue:
            traces.append(list(visited))

        queue = next_queue

    return traces

def bellman_ford_trace(G, source):
    n = G.number_of_nodes()
    dist = [float('inf')] * n
    pred = [-1] * n    
    dist[source] = 0
    pred[source] = source

    traces = []
    traces.append(list(pred))

    for _ in range(n-1):
        new_dist = list(dist)
        new_pred = list(pred)

        for u, v, data in G.edges(data=True):
            w = data['weight']

            if dist[u] + w < new_dist[v]:
                new_dist[v] = dist[u] + w
                new_pred[v] = u

            if dist[v] + w < new_dist[u]:
                new_dist[u] = dist[v] + w
                new_pred[u] = v

        if new_dist == dist:
            break

        dist = new_dist
        pred = new_pred
        traces.append(list(pred))

    return traces
