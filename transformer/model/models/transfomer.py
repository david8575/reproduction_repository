class Tranformer(nn.Moudle):
    def __init__(self, src_embed, tgt_embed, encoder, decoder, generator):
        super(Tranformer, self).__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def encode(self, src, src_mask):
        out = self.encoder(self.src_embed(src), src_mask)
        
        return out
    
    def decode(self, tgt,  encoder_out, tgt_mask, src_tgt_mask):
        out = self.decoder(self.tgt_embed(tgt), encoder_out, tgt_mask, src_tgt_mask)

        return out
    
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        src_tgt_mask = self.make_src_tgt_mask(src, tgt)
        encoder_out = self.encode(src, src_mask)
        decoder_out = self.decode(tgt, encoder_out, tgt_mask, src_tgt_mask)
        out = self.generator(decoder_out)
        out = F.log_softmax(out, dim=-1)
        
        return out, decoder_out