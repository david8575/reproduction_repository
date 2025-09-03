import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        
        self.q_fc = nn.Linear(d_model, d_model)
        self.k_fc = nn.Linear(d_model, d_model)
        self.v_fc = nn.Linear(d_model, d_model)
        self.out_fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def calculate_attention(self, query, key, value, mask):
        d_k = query.size(-1)
        attention_score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)

        attention_prob = F.softmax(attention_score, dim=-1)
        attention_prob = self.dropout(attention_prob)
        out = torch.matmul(attention_prob, value)

        return out

    def forward(self, query, key, value, mask=None):
        n_batch = query.size(0)

        def transform(x, fc):
            out = fc(x)
            out = out.view(n_batch, -1, self.h, self.d_k)
            out = out.transpose(1, 2)
            return out
        
        query = transform(query, self.q_fc)
        key = transform(key, self.k_fc)
        value = transform(value, self.v_fc)

        out = self.calculate_attention(query, key, value, mask)
        out = out.transpose(1, 2)
        out = out.contiguous().view(n_batch, -1, self.d_model)
        out = self.out_fc(out)

        return out