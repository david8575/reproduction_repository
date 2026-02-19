import torch
from torch_geometric.data import Data
from data.graph import generate_graph, graph_to_edge_index, get_edge_weights
from algorithms import bfs_trace, bellman_ford_trace, prim_trace


def create_bfs_dataset(n_graphs, n_nodes, p=0.3):
    dataset = []

    for _ in range(n_graphs):
        G = generate_graph(n_nodes, p)
        edge_index = graph_to_edge_index(G)
        edge_attr = get_edge_weights(G, edge_index) / 10.0  # 정규화: 신경망의 입력값이 0-1 범위일때 안정적
        source = 0
        traces = bfs_trace(G, source)

        for t in range(len(traces)-1):
            x = torch.tensor(traces[t], dtype=torch.float).unsqueeze(1)
            y = torch.tensor(traces[t+1], dtype=torch.float).unsqueeze(1)
            is_last = 1.0 if t == len(traces) - 2 else 0.0
            data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, is_last=is_last)
            dataset.append(data)

    return dataset


def create_bellman_ford_dataset(n_graphs, n_nodes, p=0.3):
    dataset = []

    for _ in range(n_graphs):
        G = generate_graph(n_nodes, p)
        edge_index = graph_to_edge_index(G)
        edge_attr = get_edge_weights(G, edge_index) / 10.0
        source = 0
        dist_traces, pred_traces = bellman_ford_trace(G, source)

        for t in range(len(dist_traces)-1):
            x = torch.tensor(dist_traces[t], dtype=torch.float).unsqueeze(1)
            x = torch.clamp(x, max=100.0) / 100.0
            y = torch.tensor(pred_traces[t+1], dtype=torch.long)
            mask = (y != -1)
            is_last = 1.0 if t == len(dist_traces)-2 else 0.0
            data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, mask=mask, is_last=is_last)
            dataset.append(data)

    return dataset


def create_prim_dataset(n_graphs, n_nodes, p=0.3):
    dataset = []

    for _ in range(n_graphs):
        G = generate_graph(n_nodes, p)
        edge_index = graph_to_edge_index(G)
        edge_attr = get_edge_weights(G, edge_index) / 10.0
        source = 0
        key_traces, pred_traces = prim_trace(G, source)

        for t in range(len(key_traces) - 1):
            x = torch.tensor(key_traces[t], dtype=torch.float).unsqueeze(1)
            x = torch.clamp(x, max=100.0) / 100.0
            y = torch.tensor(pred_traces[t+1], dtype=torch.long)
            mask = (y != -1)
            is_last = 1.0 if t == len(key_traces) - 2 else 0.0
            data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, mask=mask, is_last=is_last)
            dataset.append(data)

    return dataset
