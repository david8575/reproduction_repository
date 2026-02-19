import torch
import torch.nn as nn
from models.processor import Processor


# Encoder-Processor-Decoder
class AlgorithmExecutor(nn.Module):
    def __init__(self, hidden_dim=32, max_nodes=128, n_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # 알고리즘별 Encoder
        self.encoder_bfs = nn.Linear(1, hidden_dim)
        self.encoder_bf = nn.Linear(1, hidden_dim)
        self.encoder_prim = nn.Linear(1, hidden_dim)

        # 공유 Processor
        self.processor = Processor(hidden_dim)

        # 알고리즘별 Decoder
        self.decoder_bfs = nn.Linear(hidden_dim, 1)
        # BF: Q-K pointer
        self.query_bf = nn.Linear(hidden_dim, hidden_dim)
        self.key_bf = nn.Linear(hidden_dim, hidden_dim)
        # Prim: Q-K pointer
        self.query_prim = nn.Linear(hidden_dim, hidden_dim)
        self.key_prim = nn.Linear(hidden_dim, hidden_dim)

        # 알고리즘별 Termination
        self.termination_bfs = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.termination_bf = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.termination_prim = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, edge_index, edge_attr, algorithm='bfs'):
        # Encoding
        if algorithm == 'bfs':
            h = self.encoder_bfs(x)
        elif algorithm == 'bellman-ford':
            h = self.encoder_bf(x)
        elif algorithm == 'prim':
            h = self.encoder_prim(x)

        # Processing (공유)
        for _ in range(self.n_layers):
            h = self.processor(h, edge_index, edge_attr)

        # Decoding
        if algorithm == 'bfs':
            out = torch.sigmoid(self.decoder_bfs(h))
        elif algorithm == 'bellman-ford':
            q = self.query_bf(h)
            k = self.key_bf(h)
            out = torch.mm(q, k.t()) / (self.hidden_dim ** 0.5)
        elif algorithm == 'prim':
            q = self.query_prim(h)
            k = self.key_prim(h)
            out = torch.mm(q, k.t()) / (self.hidden_dim ** 0.5)

        # Termination
        if algorithm == 'bfs':
            term = self.termination_bfs(h.mean(dim=0, keepdim=True))
        elif algorithm == 'bellman-ford':
            term = self.termination_bf(h.mean(dim=0, keepdim=True))
        elif algorithm == 'prim':
            term = self.termination_prim(h.mean(dim=0, keepdim=True))

        return out, term
