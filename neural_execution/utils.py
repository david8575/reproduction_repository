"""
논문에서 다루는 그래프 알고리즘을 실제로 실행하고, 각 단계를 기록 -> 데이터셋
"""

import random
import networkx as nx

def generate_graph(n_nodes, p=0.3):
    # 연결된 그래프가 나올때까지 반복
    while True:
        # Erdős–Rényi -> n개의 노드에서 모든 노드 쌍을 p확률로 연결하는 랜덤 그래프 생성 방식(논문과 같은 방식) 
        G = nx.erdos_renyi_graph(n=n_nodes, p=p)

        if nx.is_connected(G):
            break
    
    # 각 엣지에  1-10의 랜덤 가중치를 적용(BFS에서는 이용X, Bellman-Ford에서만 이용)
    for u, v in G.edges():
        G[u][v]['weight'] = random.randint(1, 10)

    return G

# BFS -> Reachability
# BFS의 최종 해를 구하는 것인 아닌 각 스텝을 기록하여 GNN이 한 스텝을 모방하도록 학습시키기 위한 데이터셋 생성
def bfs_trace(G, source):
    n = G.number_of_nodes()

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

# Bellman-Ford -> Shortest Path
# 선행 노드를 추적하게 구현
def bellman_ford_trace(G, source):
    n = G.number_of_nodes()
    dist = [float('inf')] * n
    pred = [-1] * n    
    dist[source] = 0
    pred[source] = source

    dist_traces = []
    pred_traces = []
    dist_traces.append(list(dist))
    pred_traces.append(list(pred))

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
        dist_traces.append(list(dist))
        pred_traces.append(list(pred))

    return dist_traces, pred_traces

def prim_trace(G, source):
    n = G.number_of_nodes()
    key = [float('inf')] * n
    pred = [-1] * n
    in_mst = [False] * n

    key[source] = 0
    pred[source] = source

    key_traces = [list(key)]
    pred_traces = [list(pred)]

    for _ in range(n):
        u = min((i for i in range(n) if not in_mst[i]), key=lambda i: key[i])
        in_mst[u] = True
        
        new_key = list(key)
        new_pred = list(pred)

        for v, data in G[u].items():
            w = data['weight']

            if not in_mst[v] and w < new_key[v]:
                new_key[v] = w
                new_pred[v] = u

        if new_key == key:
            break

        key = new_key
        pred = new_pred
        key_traces.append(list(key))
        pred_traces.append(list(pred))

    return key_traces, pred_traces