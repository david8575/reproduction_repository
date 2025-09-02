class Encoder(nn.Module):
    def __init__(self, encode_layer, n_layer):
        super(Encoder, self).__init__()
        self.layers = []
        for i in range(n_layer):
            self.layers.append(copy.deepcopy(encode_layer))

    def forward(self, x, mask):
        out = x
    
        for layer in self.layers:
            out = layer(out, mask)

        return out