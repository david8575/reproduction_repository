import math
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, d_embed, vacab_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn. Embedding(vacab_size, d_embed)
        self.d_embed = d_embed

    def forward(self, x):
        out = self.embedding(x) * math.sqrt(self.d_embed)

        return out