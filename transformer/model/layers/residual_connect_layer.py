import torch.nn as nn

class ResidualConnectLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(ResidualConnectLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sub_layer):
        return x + self.dropout(sub_layer(self.layer_norm(x)))