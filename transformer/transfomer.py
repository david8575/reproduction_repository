class Tranformer(nn.Moudle):
    def __init__(self, encoder, decoder):
        super(Tranformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x, mask):
        out = self.encoder(x, mask)
        
        return out
    
    def decode(self, tgt,  encoder_out, tgt_mask):
        out = self.decode(tgt, encoder_out, tgt_mask)

        return out
    
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        encoder_out = self.encode(src, src_mask)
        y = self.decode(tgt, encoder_out, tgt_mask)

        return y
    