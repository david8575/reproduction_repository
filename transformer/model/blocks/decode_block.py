class DecoderBlock(nn.Module):
    def __init__(self, self_attetion, cross_attention, posiotion_ff):
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attetion
        self.cross_attention = cross_attention
        self.position_ff = posiotion_ff
        self.residuals = [ResidualConnectionLayer(), ResidualConnectionLayer(), ResidualConnectionLayer()]

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        out = self.residuals[0](out, lambda x: self.self_attention(query=out, key=out, value=out, mask=tgt_mask))
        out = self.residuals[1](out, lambda x: self.cross_attention(query=out, key=encoder_out, value=encoder_out, mask=src_tgt_mask))
        out = self.residuals[2](out, self.position_ff)

        return out