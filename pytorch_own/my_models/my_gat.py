import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax

"""
1. 선형 변환: 각 노드의 특성을 새로운 공간으로 변환 -> h'_i = W x h_i
    - h_i: 노드 i의 원래 특성
    - W: 학습 가능한 가중치 행렬
    - h'_i: 변환된 특성
2. 어텐션 스코어 계산: 노드 i의 입장에서 타 노드가 얼마나 중요한가를 파악 -> e_{ij} = LeakyReLU(a^t x [h'_i || h'_j])
    - [h'_i || h'_j]: 두 특성을 이어붙임(concatenate)
    - a: 학습 가능한 어텐션 벡터
    - e_ij: i→j의 어텐션 스코어 (아직 정규화 안 됨)
3. SoftMax 정규화: 어텐션 가중치의 합 1이 되도록 -> α_ij = softmax_j(e_ij) = exp(e_ij) / Σ_k exp(e_ik)
4. 가중 합계로 집계  h''_i = Σ_j α_ij x h'_j
5. 멀티 헤드 어텐션 -> 다양한 관점에서
"""

class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        # 선형 변환 가중치
        self.W = nn.Parameter(torch.Tensor(heads, in_channels, out_channels))
        # 어텐션 벡터
        self.a = nn.Parameter(torch.Tensor(heads, 2*out_channels))
        # ReLu
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a.unsqueeze(-1))

    def forward(self, x, edge_index):
        N = x.size(0)
        src, dst = edge_index
        
        # 선형 변환
        h = torch.einsum('ni,hio->nho', x, self.W)
        
        # 어텐션 스코어 계산
        h_src = h[src]
        h_dst = h[dst]
        edge_feat = torch.cat([h_src, h_dst], dim=-1)
        e = self.leaky_relu(torch.einsum('ehi,hi->eh', edge_feat, self.a))
        
        # softmax 정규화 (PyG 유틸 사용)
        alpha = softmax(e, dst, num_nodes=N)
        
        # 드롭아웃
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # 메시지 집계
        msg = alpha.unsqueeze(-1) * h_src
        out = torch.zeros(N, self.heads, self.out_channels, device=x.device)
        for h_idx in range(self.heads):
            out[:, h_idx, :].index_add_(0, dst, msg[:, h_idx, :])
        
        # 헤드 결합
        if self.concat:
            out = out.view(N, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        return out


class MyGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GATLayer(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout
        )

        self.conv2 = GATLayer(
            hidden_channels*heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout
        )

    def forward(self, x, edge_index):
        # 입력 드롭아웃
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 1 + ELU
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        # 중간 드롭아웃
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2 (출력층)
        x = self.conv2(x, edge_index)
        
        return x  # logits 반환

def get_my_gat_model(in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
    return MyGAT(
        in_channels, 
        hidden_channels, 
        out_channels, 
        heads, 
        dropout
    )
