import torch.nn as nn
import copy

class Encoder(nn.Module):
    def __init__(self, encode_layer, n_layer):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encode_layer) for _ in range(n_layer)])

    def forward(self, x, mask):
        out = x
    
        for layer in self.layers:
            out = layer(out, mask)

        return out