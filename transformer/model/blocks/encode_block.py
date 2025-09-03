import torch.nn as nn

from model.layers.residual_connect_layer import ResidualConnectLayer


class EncoderBlock(nn.Module):
    def __init__(self, self_attention, position_ff):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff
        self.residuals = [ResidualConnectLayer(), ResidualConnectLayer()]

    def forward(self, x, mask):
        out = x
        out = self.residuals[0](out, lambda out : self.self_attention(query=out, key=out, value=out, mask=mask))
        out = self.residuals[1](out, self.position_ff)

        return out
    
    