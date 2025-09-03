import torch.nn as nn

from model.layers.residual_connect_layer import ResidualConnectLayer

class DecoderBlock(nn.Module):
    def __init__(self, self_attention, cross_attention, position_ff, d_model, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.position_ff = position_ff
        self.residual1 = ResidualConnectLayer(d_model, dropout)
        self.residual2 = ResidualConnectLayer(d_model, dropout)
        self.residual3 = ResidualConnectLayer(d_model, dropout)

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        out = self.residual1(out, lambda x: self.self_attention(query=x, key=x, value=x, mask=tgt_mask))
        out = self.residual2(out, lambda x: self.cross_attention(query=x, key=encoder_out, value=encoder_out, mask=src_tgt_mask))
        out = self.residual3(out, self.position_ff)

        return out