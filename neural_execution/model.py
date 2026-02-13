import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

# Processor: 공유 GNN
class Processor(MessagePassing):
    def __init__(self, hidden_dim):
        super().__init__(aggr='max')

        self.msg_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
        )

        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        combined = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.msg_net(combined)

    def update(self, aggr_out, x):
        combined = torch.cat([x, aggr_out], dim=-1)
        return self.update_net(combined)


# Encoder-Processor-Decoder
class AlgorithmExecutor(nn.Module):
    def __init__(self, hidden_dim=32, max_nodes=128):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 알고리즘별 Encoder
        self.encoder_bfs = nn.Linear(1, hidden_dim)
        self.encoder_bf = nn.Embedding(max_nodes, hidden_dim)

        # 공유 Processor
        self.processor = Processor(hidden_dim)

        # 알고리즘별 Decoder
        self.decoder_bfs = nn.Linear(hidden_dim, 1)
        # BF: pointer mechanism (h @ h.T)이므로 별도 decoder 불필요

        # 알고리즘별 Termination
        self.termination_bfs = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.termination_bf = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, edge_index, edge_attr, algorithm='bfs'):
        # Encoding
        if algorithm == 'bfs':
            h = self.encoder_bfs(x)
        elif algorithm == 'bellman-ford':
            h = self.encoder_bf(x.clamp(min=0))  # -1(미도달)은 0으로 치환

        # Processing (공유)
        h = self.processor(h, edge_index, edge_attr)

        # Decoding
        if algorithm == 'bfs':
            out = torch.sigmoid(self.decoder_bfs(h))
        elif algorithm == 'bellman-ford':
            out = torch.mm(h, h.t())  # [n_nodes, n_nodes] pointer logits

        # Termination
        if algorithm == 'bfs':
            term = self.termination_bfs(h.mean(dim=0, keepdim=True))
        elif algorithm == 'bellman-ford':
            term = self.termination_bf(h.mean(dim=0, keepdim=True))

        return out, term
