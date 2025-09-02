import torch
import torch.nn as nn

class GraphAttentionGNN(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.relation_bias = nn.Parameter(torch.zeros(3))

    def forward(self, z, relation_matrix):
        B, N, H = z.size()
        Q = self.q_proj(z).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(z).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(z).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        bias = self.relation_bias[relation_matrix].unsqueeze(1)
        scores = scores + bias
        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, V)
        out = out.transpose(1,2).contiguous().view(B, N, H)
        out = self.out_proj(out)

        return out