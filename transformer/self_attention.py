def calculate_attention(query, key, value, mask):
    d_k = query.shape(-1)
    attention_score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        attention_score = attention_score.masked_fill(mask == 0, -1e9)

    attention_prob = F.softmax(attention_score, dim=-1)
    out = torch.matmul(attention_prob, value)

    return out
