import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from model import AlgorithmExecutor
from dataset import create_bfs_dataset, create_bellman_ford_dataset

# MPS
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cpu')
print(f"[Device] {device}")

# 학습 함수
def train_bfs_only(model, bfs_train, epochs, accum_steps=32, lr=0.001):
    # BFS 데이터만의 학습
    bce_loss = nn.BCELoss() # BFS는 이진 분류이므로 Binary Cross-Entropy 이용
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs), desc='BFS Only'):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for i, data in enumerate(bfs_train):
            data = data.to(device)
            out, term = model(data.x, data.edge_index, data.edge_attr, algorithm='bfs')
            loss_out = bce_loss(out, data.y)
            loss_term = bce_loss(term, torch.tensor([[data.is_last]]).to(device))
            loss = (loss_out + loss_term) / accum_steps
            loss.backward()
            total_loss += loss.item() * accum_steps

            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # 남은 gradient 처리
        if len(bfs_train) % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % 10 == 0:
            tqdm.write(f"    Epoch {epoch+1} | Loss {total_loss/len(bfs_train):.6f}")

def train_bf_only(model, bf_train, epochs, accum_steps=32, lr=0.001):
    # Bellman-Ford 데이터로만 학습
    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss() # 다중 분류
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs), desc='BF Only'):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for i, data in enumerate(bf_train):
            data = data.to(device)
            out, term = model(data.x, data.edge_index, data.edge_attr, algorithm='bellman-ford')
            mask = data.mask

            if mask.sum() > 0:
                loss_out = ce_loss(out[mask], data.y[mask])
            else:
                loss_out = torch.tensor(0.0).to(device)

            loss_term = bce_loss(term, torch.tensor([[data.is_last]]).to(device))
            loss = (loss_out + loss_term) / accum_steps
            loss.backward()
            total_loss += loss.item() * accum_steps

            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # 남은 gradient 처리
        if len(bf_train) % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % 10 == 0:
            tqdm.write(f"    Epoch {epoch+1} | Loss {total_loss/len(bf_train):.6f}")

def train_joint(model, bfs_train, bf_train, epochs, accum_steps=32, lr=0.001):
    # bfs bellman-ford 같이 학습
    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs), desc='BFS + BF'):
        model.train()
        total_bfs_loss = 0
        total_bf_loss = 0

        # BFS 학습
        optimizer.zero_grad()
        for i, data in enumerate(bfs_train):
            data = data.to(device)
            out, term = model(data.x, data.edge_index, data.edge_attr, algorithm='bfs')
            loss_out = bce_loss(out, data.y)
            loss_term = bce_loss(term, torch.tensor([[data.is_last]]).to(device))
            loss = (loss_out + loss_term) / accum_steps
            loss.backward()
            total_bfs_loss += loss.item() * accum_steps

            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        if len(bfs_train) % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        # BF 학습
        optimizer.zero_grad()
        for i, data in enumerate(bf_train):
            data = data.to(device)
            out, term = model(data.x, data.edge_index, data.edge_attr, algorithm='bellman-ford')
            mask = data.mask

            if mask.sum() > 0:
                loss_out = ce_loss(out[mask], data.y[mask])
            else:
                loss_out = torch.tensor(0.0).to(device)

            loss_term = bce_loss(term, torch.tensor([[data.is_last]]).to(device))
            loss = (loss_out + loss_term) / accum_steps
            loss.backward()
            total_bf_loss += loss.item() * accum_steps

            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        if len(bf_train) % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % 10 == 0:
            avg_bfs = total_bfs_loss / len(bfs_train)
            avg_bf = total_bf_loss / len(bf_train)
            tqdm.write(f"    Epoch {epoch+1} | BFS Loss {avg_bfs:.6f} | BF Loss {avg_bf:.6f}")

# 평가 함수
def evaluate_bfs(model, n_nodes_list, n_graphs_test=20):
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


def evaluate_bf(model, n_nodes_list, n_graphs_test=20):
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

# 실행 함수
def mode_train(args):
    print(f"[Dataset Generating......]")
    bfs_train = create_bfs_dataset(n_graphs=args.n_graphs, n_nodes=args.n_nodes)
    bf_train = create_bellman_ford_dataset(n_graphs=args.n_graphs, n_nodes=args.n_nodes)
    print(f"[BFS] Train {len(bfs_train)} | [Bellman-Ford] Train {len(bf_train)}")

    model = AlgorithmExecutor(hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
    train_joint(model, bfs_train, bf_train, args.epochs, accum_steps=args.accum_steps, lr=args.lr)

    n_nodes_list = args.test_nodes

    print(f"[Test]")
    bfs_results = evaluate_bfs(model, n_nodes_list, n_graphs_test=args.n_graphs_test)
    bf_results = evaluate_bf(model, n_nodes_list, n_graphs_test=args.n_graphs_test)

    for n in n_nodes_list:
        print(f"    [{n:2d} Nodes] BFS Acc: {bfs_results[n]:.1f}% | BF Acc: {bf_results[n]:.1f}%")

def mode_transfer(args):
    print(f"[Dataset Generating......]")
    bfs_train = create_bfs_dataset(n_graphs=args.n_graphs, n_nodes=args.n_nodes)
    bf_train = create_bellman_ford_dataset(n_graphs=args.n_graphs, n_nodes=args.n_nodes)

    n_nodes_list = args.test_nodes

    # BF 단독
    print(f"[BF Only]")
    model_single = AlgorithmExecutor(hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
    train_bf_only(model_single, bf_train, args.epochs, accum_steps=args.accum_steps, lr=args.lr)
    results_single = evaluate_bf(model_single, n_nodes_list, n_graphs_test=args.n_graphs_test)

    # BFS + BF 동시
    print(f"[BFS + BF Joint]")
    model_joint = AlgorithmExecutor(hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
    train_joint(model_joint, bfs_train, bf_train, args.epochs, accum_steps=args.accum_steps, lr=args.lr)
    results_joint = evaluate_bf(model_joint, n_nodes_list, n_graphs_test=args.n_graphs_test)

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
                        default=128)
    parser.add_argument("--n_graphs",
                        type=int,
                        default=500)
    parser.add_argument("--n_nodes",
                        type=int,
                        default=20)
    parser.add_argument("--test_nodes",
                        type=int,
                        nargs='+',
                        default=[20, 50, 100])
    parser.add_argument("--n_graphs_test",
                        type=int,
                        default=20)
    parser.add_argument("--accum_steps",
                        type=int,
                        default=32)
    parser.add_argument("--lr",
                        type=float,
                        default=0.001)
    parser.add_argument("--n_layers",
                        type=int,
                        default=3)
    args = parser.parse_args()

    print("=" * 55)
    print("[Experiment Settings]")
    print("=" * 55)
    for k, v in vars(args).items():
        print(f"  {k:>15}: {v}")
    print("=" * 55)

    if args.mode == "train":
        mode_train(args)
    elif args.mode == "transfer":
        mode_transfer(args)
