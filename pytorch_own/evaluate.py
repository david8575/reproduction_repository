import argparse
import torch
from data_loader import load_data
from train import get_model, test


def parse_args():
    parser = argparse.ArgumentParser(description='model performance comparison')
    parser.add_argument('--data', type=str, default='Cora')
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['pyg_gat', 'my_gat'],
                        help='models compared [pyg_gat my_gat]')
    parser.add_argument('--hidden', type=int, default=8)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.6)
    return parser.parse_args()


def main():
    args = parse_args()
    
    data, num_classes = load_data(args.data)
    
    results = {}
    
    print("\n" + "="*50)
    print(f"{'MODEL':<15} {'TEST ACC':>10}")
    print("="*50)
    
    for model_name in args.models:
        model = get_model(
            model_name,
            in_channels=data.num_features,
            hidden_channels=args.hidden,
            out_channels=num_classes,
            heads=args.heads,
            dropout=args.dropout
        )
        
        try:
            model.load_state_dict(torch.load(f'checkpoints/{model_name}_best.pt'))
            accs = test(model, data)
            results[model_name] = accs['test']
            print(f"{model_name:<15} {accs['test']:>10.4f}")
        except FileNotFoundError:
            print(f"{model_name:<15} {'(not trained)':>10}")
    
    print("="*50)
    
    if results:
        best_model = max(results, key=results.get)
        print(f"\nBest: {best_model} ({results[best_model]:.4f})")


if __name__ == '__main__':
    main()
