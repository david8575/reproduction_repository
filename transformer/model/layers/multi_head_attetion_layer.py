class MultiHeadAttetionLayer(nn.Mudule):
    def __init__(self, d_model, h, qkv_fc, out_fc):
        super(MultiHeadAttetionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.qkv_fc = qkv_fc
        self.q_fc = copy.deepcopy(qkv_fc)
        self.k_fc = copy.deepcopy(qkv_fc)
        self.v_fc = copy.deepcopy(qkv_fc)
        self.out_fc = out_fc

    def calculate_attention(query, key, value, mask):
        d_k = query.shape(-1)
        attention_score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)

        attention_prob = F.softmax(attention_score, dim=-1)
        out = torch.matmul(attention_prob, value)

        return out


    def forward(self, *args, query, key, value, mask=None):
        n_batch = query.size(0)

        def transform(x, fc):
            out = fc(x)
            out = out.view(n_batch, -1, self.h, self.d_model // self.h)
            out = out.transpose(1, 2)

            return out
        
        query = transform(query, self.q_fc)
        key = transform(key, self.k_fc)
        value = transform(value, self.v_fc)

        out = calculate_attention(query, key, value, mask)
        out = out.transpose(1,2)
        out = out.contiguous().view(n_batch, -1, self.d_model)
        out = self.out_fc(out)

        return out