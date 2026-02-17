import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

# Processor: 공유 GNN
class Processor(MessagePassing):
    def __init__(self, hidden_dim):
        # 논문 실험 결과 가장 좋은 성능을 보인 메세지패싱의 max를 이용
        # BFS에서는 이웃 중 하나라도 방문됐다면 나도 방문 -> max로 자연스러운 표현
        # BF에서는 이웃 중 최소 거리를 선택하는 부분에서 부호를 반대로 바꾸어 자연스러운 max표현
        super().__init__(aggr='max')

        self.msg_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
        )

        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

    """
    - Message Passing
        1. message
            1) 각 엣지(j->i)에 대해, 노드 i의 상태(x_i), 노드j의 상태(x_j), 엣지 가중치(edge_attr)를 합쳐서 메세지를 생성
            2) 입력 차원: hidden_dim * 2 + 1(노드 2+ 엣지 가중치 1)
        2. aggr(max)
            1) 노드 i로 들어오는 모든 메세지 중 최대값을 선택
        3. update
            1) 기존 상태 + 집계된 메세지를 합쳐서 새 상태 생성
    """

    def message(self, x_i, x_j, edge_attr):
        combined = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.msg_net(combined)

    def update(self, aggr_out, x):
        combined = torch.cat([x, aggr_out], dim=-1)
        return self.update_net(combined)
    
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)


# Encoder-Processor-Decoder
class AlgorithmExecutor(nn.Module):
    def __init__(self, hidden_dim=32, max_nodes=128, n_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # 알고리즘별 Encoder
        self.encoder_bfs = nn.Linear(1, hidden_dim)
        self.encoder_bf = nn.Linear(1, hidden_dim)

        # 공유 Processor
        self.processor = Processor(hidden_dim)

        # 알고리즘별 Decoder
        self.decoder_bfs = nn.Linear(hidden_dim, 1)
        # BF: Q-K pointer
        self.query_bf = nn.Linear(hidden_dim, hidden_dim)
        self.key_bf = nn.Linear(hidden_dim, hidden_dim)

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
            h = self.encoder_bf(x) 

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

        # Termination
        if algorithm == 'bfs':
            term = self.termination_bfs(h.mean(dim=0, keepdim=True))
        elif algorithm == 'bellman-ford':
            term = self.termination_bf(h.mean(dim=0, keepdim=True))

        return out, term
