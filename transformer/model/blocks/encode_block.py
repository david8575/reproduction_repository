import torch.nn as nn

from model.layers.residual_connect_layer import ResidualConnectLayer

class EncoderBlock(nn.Module):
    def __init__(self, self_attention, position_ff, d_model, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff
        self.residual1 = ResidualConnectLayer(d_model, dropout)
        self.residual2 = ResidualConnectLayer(d_model, dropout)

    def forward(self, x, mask):
        out = self.residual1(x, lambda x: self.self_attention(query=x, key=x, value=x, mask=mask))
        out = self.residual2(out, self.position_ff)

        return out