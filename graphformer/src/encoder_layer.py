import torch
import torch.nn as nn
from .gnn import GraphAttentionGNN

class GraphFormerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.gnn = GraphAttentionGNN(hidden_size, num_heads)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, H_g, relation_matrix):
        B, N, T, H = H_g.size()
        cls_tokens = H_g[:, :, 0, :]  
        z_hat = self.gnn(cls_tokens, relation_matrix)  

        H_aug = torch.cat([z_hat.unsqueeze(2), H_g], dim=2)
        H_aug = H_aug.view(B * N, T + 1, H)

        residual = H_aug
        H_aug, _ = self.attn(H_aug, H_aug, H_aug)
        H_aug = self.norm1(H_aug + residual)

        residual = H_aug
        H_aug = self.ffn(H_aug)
        H_aug = self.norm2(H_aug + residual)

        return H_aug.view(B, N, T + 1, H)
