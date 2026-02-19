import argparse
import torch
import torch.nn.functional as F
from  data_loader import load_data

def parse_args():
    parser = argparse.ArgumentParser(description='GNN Training')

    # 모델 선택
    parser.add_argument("--model", type=str, required=True, help='choose model[pyg_gat, my_gat]')

    # 데이터셋 선택
    parser.add_argument('--data', type=str, default='Cora', choices=['Cora', 'CiteSeer', 'PubMed'], help='choose dataset')

    # 하이퍼 파라미터
    parser.add_argument('--hidden', type=int, default=8)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()

def get_model(model_name, in_channels, hidden_channels, out_channels, heads, dropout):
    # === GAT ===
    if model_name == 'pyg_gat':
        from original_models.pyg_gat import get_pyg_gat_model
        return get_pyg_gat_model(in_channels, hidden_channels, out_channels, heads, dropout)
    
    elif model_name == 'my_gat':
        from my_models.my_gat import get_my_gat_model
        return get_my_gat_model(in_channels, hidden_channels, out_channels, heads, dropout)
    
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    accs = {}
    for split in ['train', 'val', 'test']:
        mask = getattr(data, f'{split}_mask')
        accs[split] = (pred[mask] == data.y[mask]).float().mean().item()
    
    return accs

def main():
    args = parse_args()
    
    # 시드 고정
    torch.manual_seed(args.seed)
    
    # 데이터 로드
    data, num_classes = load_data(args.data)
    
    # 모델 생성
    model = get_model(
        args.model,
        in_channels=data.num_features,
        hidden_channels=args.hidden,
        out_channels=num_classes,
        heads=args.heads,
        dropout=args.dropout
    )
    
    print(f"\n[Model]: {args.model}")
    print(f"[Params]: {sum(p.numel() for p in model.parameters()):,}")
    
    # 옵티마이저
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # 학습 루프
    best_val_acc = 0
    for epoch in range(1, args.epochs + 1):
        loss = train(model, data, optimizer)
        accs = test(model, data)
        
        if accs['val'] > best_val_acc:
            best_val_acc = accs['val']
            torch.save(model.state_dict(), f'checkpoints/{args.model}_best.pt')
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                  f"Train: {accs['train']:.4f} | Val: {accs['val']:.4f} | Test: {accs['test']:.4f}")
    
    # 최종 결과
    model.load_state_dict(torch.load(f'checkpoints/{args.model}_best.pt'))
    final = test(model, data)
    print(f"\n=== {args.model} Final Test Acc: {final['test']:.4f} ===")
    
    return final['test']


if __name__ == '__main__':
    main()
