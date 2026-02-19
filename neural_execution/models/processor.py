import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


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
            2) 입력 차원: hidden_dim * 2 + 1(노드 2 + 엣지 가중치 1)
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
