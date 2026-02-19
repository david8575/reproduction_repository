import argparse
import torch
from models import AlgorithmExecutor
from data import create_bfs_dataset, create_bellman_ford_dataset, create_prim_dataset
from trainer import train_bf_only, train_prim_only, train_joint
from trainer import evaluate_bfs, evaluate_bf, evaluate_prim

device = torch.device('cpu')
print(f"[Device] {device}")


def mode_train(args):
    print(f"[Dataset Generating......]")
    bfs_train = create_bfs_dataset(n_graphs=args.n_graphs, n_nodes=args.n_nodes)
    bf_train = create_bellman_ford_dataset(n_graphs=args.n_graphs, n_nodes=args.n_nodes)
    prim_train = create_prim_dataset(n_graphs=args.n_graphs, n_nodes=args.n_nodes)
    print(f"[BFS] Train {len(bfs_train)} | [BF] Train {len(bf_train)} | [Prim] Train {len(prim_train)}")

    model = AlgorithmExecutor(hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
    train_joint(model, bfs_train, bf_train, prim_train, args.epochs, device,
                accum_steps=args.accum_steps, lr=args.lr)

    n_nodes_list = args.test_nodes

    print(f"[Test]")
    bfs_results = evaluate_bfs(model, n_nodes_list, device, n_graphs_test=args.n_graphs_test)
    bf_results = evaluate_bf(model, n_nodes_list, device, n_graphs_test=args.n_graphs_test)
    prim_results = evaluate_prim(model, n_nodes_list, device, n_graphs_test=args.n_graphs_test)

    print(f"{'Nodes':>8} | {'BFS Acc':>10} | {'BF Acc':>10} | {'Prim Acc':>10}")
    print("-" * 50)
    for n in n_nodes_list:
        print(f"{n:>8} | {bfs_results[n]:>9.1f}% | {bf_results[n]:>9.1f}% | {prim_results[n]:>9.1f}%")


def mode_transfer(args):
    print(f"[Dataset Generating......]")
    bfs_train = create_bfs_dataset(n_graphs=args.n_graphs, n_nodes=args.n_nodes)
    bf_train = create_bellman_ford_dataset(n_graphs=args.n_graphs, n_nodes=args.n_nodes)
    prim_train = create_prim_dataset(n_graphs=args.n_graphs, n_nodes=args.n_nodes)

    n_nodes_list = args.test_nodes

    # BF 단독
    print(f"[BF Only]")
    model_bf_single = AlgorithmExecutor(hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
    train_bf_only(model_bf_single, bf_train, args.epochs, device,
                  accum_steps=args.accum_steps, lr=args.lr)
    results_bf_single = evaluate_bf(model_bf_single, n_nodes_list, device,
                                    n_graphs_test=args.n_graphs_test)

    # Prim 단독
    print(f"[Prim Only]")
    model_prim_single = AlgorithmExecutor(hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
    train_prim_only(model_prim_single, prim_train, args.epochs, device,
                    accum_steps=args.accum_steps, lr=args.lr)
    results_prim_single = evaluate_prim(model_prim_single, n_nodes_list, device,
                                        n_graphs_test=args.n_graphs_test)

    # BFS + BF + Prim 동시
    print(f"[BFS + BF + Prim Joint]")
    model_joint = AlgorithmExecutor(hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
    train_joint(model_joint, bfs_train, bf_train, prim_train, args.epochs, device,
                accum_steps=args.accum_steps, lr=args.lr)
    results_bf_joint = evaluate_bf(model_joint, n_nodes_list, device,
                                   n_graphs_test=args.n_graphs_test)
    results_prim_joint = evaluate_prim(model_joint, n_nodes_list, device,
                                       n_graphs_test=args.n_graphs_test)

    # BF 전이 비교
    print("\n" + "=" * 59)
    print(f"[Transfer Learning Comparison (BF Accuracy %)]")
    print("=" * 59)
    print(f"{'Nodes':>8} | {'BF Only':>10} | {'BFS+BF+Prim':>13} | {'diff':>8}")
    print("-" * 59)
    for n in n_nodes_list:
        s = results_bf_single[n]
        j = results_bf_joint[n]
        diff = j - s
        sign = "+" if diff >= 0 else ""
        print(f"{n:>8} | {s:>9.1f}% | {j:>12.1f}% | {sign}{diff:>6.1f}%")

    # Prim 전이 비교
    print("\n" + "=" * 59)
    print(f"[Transfer Learning Comparison (Prim Accuracy %)]")
    print("=" * 59)
    print(f"{'Nodes':>8} | {'Prim Only':>10} | {'BFS+BF+Prim':>13} | {'diff':>8}")
    print("-" * 59)
    for n in n_nodes_list:
        s = results_prim_single[n]
        j = results_prim_joint[n]
        diff = j - s
        sign = "+" if diff >= 0 else ""
        print(f"{n:>8} | {s:>9.1f}% | {j:>12.1f}% | {sign}{diff:>6.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Execution of Graph Algorithms")
    parser.add_argument("--mode",
                        type=str,
                        default="train",
                        choices=["train", "transfer"],
                        help="train: BFS+BF+Prim joint | transfer: solo vs joint 비교")
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
