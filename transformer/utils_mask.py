def make_pad_mask(self, query, key, pad_idx=1):
    query_seq_len, key_seq_len = query.size(1), key.size(1)

    key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)
    key_mask = key_mask.repeat(1,1, query_seq_len, 1)

    query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)
    query_mask = query_mask.repeat(1, 1, 1, key_seq_len)

    mask = key_mask & query_mask
    mask.requires_grad = False

    return mask

def make_src_mask(self, src):
    pad_mask = self.make_pad_mask(src, src)

    return pad_mask

def make_subsequent_mask(query, key):
    query_seq_len, key_seq_len = query.size(1), key.size(1)

    tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype('uint8')
    mask = torch.tnesor(tril, dtype=torch.bool, requires_grad=False, device=query.device)

    return mask

def make_tgt_mask(self, tgt):
    pad_mask = self.make_pad_mask(tgt, tgt)
    subsequent_mask = self.make_subsequent_mask(tgt, tgt)
    mask = pad_mask & subsequent_mask

    return subsequent_mask