import torch
from torch_geometric.data import Data
from utils import generate_graph, bfs_trace

def graph_to_edge_index(G):
    """
    networkx graph -> PyG edge_index
    """
    edges = list(G.edges())
    source = [e[0] for e in edges] + [e[1] for e in edges]
    target = [e[1] for e in edges] + [e[0] for e in edges]
    edge_index = torch.tensor([source, target], dtype=torch.long)

    return edge_index

def create_bfs_dataset(n_graphs, n_nodes, p=0.3):
    """
    generate BFS dataset
    """
    dataset= []

    for _ in range(n_graphs):
        G = generate_graph(n_nodes, p)
        edge_index = graph_to_edge_index(G)
        source = 0
        traces = bfs_trace(G, source)

        # 연속된 두 단계를 (입력, 정답) 쌍으로 만들기
        for t in range(len(traces)-1):
            x = torch.tensor(traces[t], dtype=torch.float).unsqueeze(1)
            y = torch.tensor(traces[t+1], dtype=torch.float).unsqueeze(1)
            data = Data(x=x, y=y, edge_index=edge_index)
            dataset.append(data)

    return dataset

if __name__=="__main__":
    dataset = create_bfs_dataset(n_graphs=5, n_nodes=8)
    print(f"총 샘플 수: {len(dataset)}")
    print(f"\n첫 번째 샘플:")
    print(f"  x (입력):  {dataset[0].x.squeeze().tolist()}")
    print(f"  y (정답):  {dataset[0].y.squeeze().tolist()}")
    print(f"  edge_index shape: {dataset[0].edge_index.shape}")