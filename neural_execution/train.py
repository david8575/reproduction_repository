import argparse
import torch
import torch.nn as nn
from model import AlgorithmExecutor
from dataset import create_bfs_dataset, create_bellman_ford_dataset

# 학습 함수
def train_bfs_only(model, bfs_train, epochs):
    # BFS 데이터만의 학습
    bce_loss = nn.BCELoss() # BFS는 이진 분류이므로 Binary Cross-Entropy 이용
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for data in bfs_train:
            optimizer.zero_grad()
            out, term = model(data.x, data.edge_index, data.edge_attr, algorithm='bfs')
            loss_out = bce_loss(out, data.y)
            loss_term = bce_loss(term, torch.tensor([[data.is_last]]))
            loss = loss_out + loss_term
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1} | Loss {total_loss/len(bfs_train):.6f}")

def train_bf_only(model, bf_train, epochs):
    # Bellman-Ford 데이터로만 학습
    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss() # 다중 분류
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for data in bf_train:
            optimizer.zero_grad()
            out, term = model(data.x, data.edge_index, data.edge_attr, algorithm='bellman-ford')
            mask = data.mask
            if mask.sum() > 0:
                loss_out = ce_loss(out[mask], data.y[mask])
            else:
                loss_out = torch.tensor(0.0)
            loss_term = bce_loss(term, torch.tensor([[data.is_last]]))
            loss = loss_out + loss_term
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1} | Loss {total_loss/len(bf_train):.6f}")

def train_joint(model, bfs_train, bf_train, epochs):
    # bfs bellman-ford 같이 학습
    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_bfs_loss = 0
        total_bf_loss = 0

        for data in bfs_train:
            optimizer.zero_grad()
            out, term = model(data.x, data.edge_index, data.edge_attr, algorithm='bfs')
            loss_out = bce_loss(out, data.y)
            loss_term = bce_loss(term, torch.tensor([[data.is_last]]))
            loss = loss_out + loss_term
            loss.backward()
            optimizer.step()
            total_bfs_loss += loss.item()

        for data in bf_train:
            optimizer.zero_grad()
            out, term = model(data.x, data.edge_index, data.edge_attr, algorithm='bellman-ford')
            mask = data.mask
            if mask.sum() > 0:
                loss_out = ce_loss(out[mask], data.y[mask])
            else:
                loss_out = torch.tensor(0.0)
            loss_term = bce_loss(term, torch.tensor([[data.is_last]]))
            loss = loss_out + loss_term
            loss.backward()
            optimizer.step()
            total_bf_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_bfs = total_bfs_loss / len(bfs_train)
            avg_bf = total_bf_loss / len(bf_train)
            print(f"    Epoch {epoch+1} | BFS Loss {avg_bfs:.6f} | BF Loss {avg_bf:.6f}")

# 평가 함수
def evaluate_bfs(model, n_nodes_list):
    model.eval()
    results = {}

    for n_nodes in n_nodes_list:
        test_data = create_bfs_dataset(n_graphs=20, n_nodes=n_nodes)
        correct, total = 0, 0

        with torch.no_grad():
            for data in test_data:
                out, _ = model(data.x, data.edge_index, data.edge_attr, algorithm='bfs')
                pred = (out > 0.5).float()
                correct += (pred == data.y).sum().item()
                total += data.y.numel()

        results[n_nodes] = correct / total * 100

    return results


def evaluate_bf(model, n_nodes_list):
    model.eval()
    results = {}

    for n_nodes in n_nodes_list:
        test_data = create_bellman_ford_dataset(n_graphs=20, n_nodes=n_nodes)
        correct, total = 0, 0

        with torch.no_grad():
            for data in test_data:
                out, _ = model(data.x, data.edge_index, data.edge_attr, algorithm='bellman-ford')
                mask = data.mask
                if mask.sum() > 0:
                    pred = out[mask].argmax(dim=1)
                    correct += (pred == data.y[mask]).sum().item()
                    total += mask.sum().item()

        results[n_nodes] = correct / total * 100 if total > 0 else 0

    return results

# 실행 함수
def mode_train(args):
    print(f"[Dataset Generating......]")
    bfs_train = create_bfs_dataset(n_graphs=500, n_nodes=20)
    bf_train = create_bellman_ford_dataset(n_graphs=500, n_nodes=20)
    print(f"[BFS] Train {len(bfs_train)} | [Bellman-Ford] Train {len(bf_train)}")

    model = AlgorithmExecutor(hidden_dim=args.hidden_dim)
    train_joint(model, bfs_train, bf_train, args.epochs)

    n_nodes_list = [20, 50, 100]
    
    print(f"[Test]")
    bfs_results = evaluate_bfs(model, n_nodes_list)
    bf_results = evaluate_bf(model, n_nodes_list)

    for n in n_nodes_list:
        print(f"    [{n:2d} Nodes] BFS Acc: {bfs_results[n]:.1f}% | BF Acc: {bf_results[n]:.1f}%")

def mode_transfer(args):
    print(f"[Dataset Generating......]")
    bfs_train = create_bfs_dataset(n_graphs=500, n_nodes=20)
    bf_train = create_bellman_ford_dataset(n_graphs=500, n_nodes=20)

    n_nodes_list = [20, 50, 100]

    # BF 단독
    print(f"[BF Only]")
    model_single = AlgorithmExecutor(hidden_dim=args.hidden_dim)
    train_bf_only(model_single, bf_train, args.epochs)
    results_single = evaluate_bf(model_single, n_nodes_list)

    # BFS + BF 동시
    print(f"[BFS + BF Joint]")
    model_joint = AlgorithmExecutor(hidden_dim=args.hidden_dim)
    train_joint(model_joint, bfs_train, bf_train, args.epochs)
    results_joint = evaluate_bf(model_joint, n_nodes_list)

    # 결과 비교
    print("\n" + "=" * 55)
    print(f"[Transfer Learning Comparison (BF Accuracy %)]")
    print("=" * 55)
    print(f"{'Nodes':>8} | {'BF Only':>10} | {'BFS + BF':>12} | {'diff':>8}")
    print("-" * 55)
    for n in n_nodes_list:
        s = results_single[n]
        j = results_joint[n]
        diff = j - s
        sign = "+" if diff >= 0 else ""
        print(f"{n:>8} | {s:>9.1f}% | {j:>11.1f}% | {sign}{diff:>6.1f}%")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Neural Execution of Graph Algorithms")
    parser.add_argument("--mode", 
                        type=str, 
                        default="train", 
                        choices=["train", "transfer"],
                        help="train: BFS+Bellman-Ford | transfer: 1) BFS 2) BFS+Bellman-Ford")
    parser.add_argument("--epochs",
                        type=int,
                        default=500)
    parser.add_argument("--hidden_dim",
                        type=int,
                        default=64)
    args = parser.parse_args()

    if args.mode == "train":
        mode_train(args)
    elif args.mode == "transfer":
        mode_transfer(args)
