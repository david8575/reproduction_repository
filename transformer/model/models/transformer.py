import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Transformer(nn.Module):
    def __init__(self, src_embed, tgt_embed, encoder, decoder, generator):
        super().__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def encode(self, src, src_mask):
        out = self.encoder(self.src_embed(src), src_mask)
        return out
    
    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
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

    def make_pad_mask(self, query, key, pad_idx=0):
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)
        key_mask = key_mask.repeat(1, 1, query_seq_len, 1)

        query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)
        query_mask = query_mask.repeat(1, 1, 1, key_seq_len)

        mask = key_mask & query_mask
        mask.requires_grad = False

        return mask

    def make_src_mask(self, src):
        pad_mask = self.make_pad_mask(src, src)
        return pad_mask

    def make_subsequent_mask(self, query, key):
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        tril = torch.tril(torch.ones((query_seq_len, key_seq_len), dtype=torch.bool, device=query.device))
        return tril

    def make_tgt_mask(self, tgt):
        pad_mask = self.make_pad_mask(tgt, tgt)
        subsequent_mask = self.make_subsequent_mask(tgt, tgt)
        mask = pad_mask & subsequent_mask
        return mask

    def make_src_tgt_mask(self, src, tgt):
        pad_mask = self.make_pad_mask(tgt, src)
        return pad_mask