import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class MPNN(MessagePassing):
    def __init__(self, hidden_dim=32):
        super().__init__(aggr='max')

        # 메세지 생성 네트워크: 이웃 feature -> 메세지
        self.msg_net = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
        )

        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        self.encoder = nn.Linear(1, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        h = self.encoder(x)
        h = self.propagate(edge_index, x=h)
        out = self.decoder(h)

        return torch.sigmoid(out)
    
    def message(self, x_j):
        return self.msg_net(x_j)
    
    def update(self, aggr_out, x):
        combined = torch.cat([x, aggr_out], dim=-1)
        
        return self.update_net(combined)
    
if __name__ == "__main__":
    from dataset import create_bfs_dataset

    model = MPNN(hidden_dim=32)
    dataset = create_bfs_dataset(n_graphs=3, n_nodes=8)

    data = dataset[0]
    out = model(data.x, data.edge_index)

    print(f"입력 shape:  {data.x.shape}")       # [8, 1]
    print(f"출력 shape:  {out.shape}")           # [8, 1]
    print(f"입력 (현재 상태): {data.x.squeeze().tolist()}")
    print(f"출력 (예측):      {[round(v, 2) for v in out.squeeze().tolist()]}")
    print(f"정답 (다음 상태): {data.y.squeeze().tolist()}")