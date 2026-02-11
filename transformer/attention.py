import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask==True, float('-inf'))

    attention_weight = F.softmax(scores, dim=-1)

    if dropout is not None:
        attention_weight = dropout(attention_weight)

    output = torch.matmul(attention_weight, value)

    return output, attention_weight

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)


        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)

        attention_output, attention_weights = scaled_dot_product_attention(q,k,v,mask=mask, dropout=self.dropout)

        attention_output = attention_output.transpose(1,2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.d_model)

        output = self.w_o(attention_output)

        return output, attention_weights