import torch
from data import create_bfs_dataset, create_bellman_ford_dataset, create_prim_dataset


def evaluate_bfs(model, n_nodes_list, device, n_graphs_test=20):
    model.eval()
    results = {}

    for n_nodes in n_nodes_list:
        test_data = create_bfs_dataset(n_graphs=n_graphs_test, n_nodes=n_nodes)
        correct, total = 0, 0

        with torch.no_grad():
            for data in test_data:
                data = data.to(device)
                out, _ = model(data.x, data.edge_index, data.edge_attr, algorithm='bfs')
                pred = (out > 0.5).float()
                correct += (pred == data.y).sum().item()
                total += data.y.numel()

        results[n_nodes] = correct / total * 100

    return results


def evaluate_bf(model, n_nodes_list, device, n_graphs_test=20):
    model.eval()
    results = {}

    for n_nodes in n_nodes_list:
        test_data = create_bellman_ford_dataset(n_graphs=n_graphs_test, n_nodes=n_nodes)
        correct, total = 0, 0

        with torch.no_grad():
            for data in test_data:
                data = data.to(device)
                out, _ = model(data.x, data.edge_index, data.edge_attr, algorithm='bellman-ford')
                mask = data.mask

                if mask.sum() > 0:
                    pred = out[mask].argmax(dim=1)
                    correct += (pred == data.y[mask]).sum().item()
                    total += mask.sum().item()

        results[n_nodes] = correct / total * 100 if total > 0 else 0

    return results


def evaluate_prim(model, n_nodes_list, device, n_graphs_test=20):
    model.eval()
    results = {}

    for n_nodes in n_nodes_list:
        test_data = create_prim_dataset(n_graphs=n_graphs_test, n_nodes=n_nodes)
        correct, total = 0, 0

        with torch.no_grad():
            for data in test_data:
                data = data.to(device)
                out, _ = model(data.x, data.edge_index, data.edge_attr, algorithm='prim')
                mask = data.mask

                if mask.sum() > 0:
                    pred = out[mask].argmax(dim=1)
                    correct += (pred == data.y[mask]).sum().item()
                    total += mask.sum().item()

        results[n_nodes] = correct / total * 100 if total > 0 else 0

    return results
