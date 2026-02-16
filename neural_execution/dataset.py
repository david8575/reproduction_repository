import torch
from torch_geometric.data import Data
from utils import generate_graph, bfs_trace, bellman_ford_trace

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

def create_bfs_dataset(n_graphs, n_nodes, p=0.3):
    dataset = []

    for _ in range(n_graphs):
        G = generate_graph(n_nodes, p)
        edge_index = graph_to_edge_index(G)
        edge_attr = get_edge_weights(G, edge_index) / 10.0  # 정규화 신경망의 입력값이 0-1범위일때 안정적이기 때문
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

            mask = (y!=-1)
            is_last = 1.0 if t == len(dist_traces)-2 else 0.0
            data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, mask=mask, is_last=is_last)
            dataset.append(data)
            
    return dataset


if __name__=="__main__":
    bfs_dataset = create_bfs_dataset(n_graphs=5, n_nodes=8)
    print(f"    [BFS Samples] {len(bfs_dataset)}")
    print(f"        x: {bfs_dataset[0].x.squeeze().tolist()}")
    print(f"        y: {bfs_dataset[0].y.squeeze().tolist()}")

    bf_dataset = create_bellman_ford_dataset(n_graphs=3, n_nodes=6)
    print(f"    [BF Samples] {len(bf_dataset)}")
    print(f"        x (predecessor): {bf_dataset[0].x.tolist()}")
    print(f"        y (predecessor): {bf_dataset[0].y.tolist()}")
    print(f"        mask: {bf_dataset[0].mask.tolist()}")
