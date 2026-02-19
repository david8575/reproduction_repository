import random
import torch
import networkx as nx


def generate_graph(n_nodes, p=0.3):
    # 연결된 그래프가 나올때까지 반복
    while True:
        # Erdős–Rényi -> n개의 노드에서 모든 노드 쌍을 p확률로 연결하는 랜덤 그래프 생성 방식(논문과 같은 방식)
        G = nx.erdos_renyi_graph(n=n_nodes, p=p)

        if nx.is_connected(G):
            break

    # 각 엣지에 1-10의 랜덤 가중치를 적용(BFS에서는 이용X, Bellman-Ford/Prim에서만 이용)
    for u, v in G.edges():
        G[u][v]['weight'] = random.randint(1, 10)

    return G


def graph_to_edge_index(G):
    # networkx graph -> PyG edge_index
    edges = list(G.edges())
    source = [e[0] for e in edges] + [e[1] for e in edges]
    target = [e[1] for e in edges] + [e[0] for e in edges]
    edge_index = torch.tensor([source, target], dtype=torch.long)

    return edge_index


def get_edge_weights(G, edge_index):
    weights = []

    for i in range(edge_index.shape[1]):
        u = edge_index[0][i].item()
        v = edge_index[1][i].item()
        weights.append(G[u][v]['weight'])

    return torch.tensor(weights, dtype=torch.float).unsqueeze(1)
