import torch
import torch.nn as nn
from model import MPNN
from dataset import create_bfs_dataset

train_dataset = create_bfs_dataset(n_graphs=1000, n_nodes=16)
test_dataset = create_bfs_dataset(n_graphs=100, n_nodes=12)
print(f"[Training Sample]: {len(train_dataset)} [Testing Sample]: {len(test_dataset)}")

model = MPNN(hidden_dim=32)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(200):
    model.train()
    total_loss = 0

    for data in train_dataset:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch+1) % 20 == 0:
        avg_loss = total_loss / len(train_dataset)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_dataset:
                out = model(data.x, data.edge_index)
                pred = (out > 0.5).float()
                correct += (pred == data.y).sum().item()
                total += data.y.numel()

        acc = correct / total * 100
        print(f"[Epoch {epoch+1:3d}]: Loss: {avg_loss:.4f} | Test Acc: {acc:.4f}%")