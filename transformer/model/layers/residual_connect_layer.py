import torch.nn as nn

class ResidualConnectLayer(nn.Module):
    def __init__(self):
        super(ResidualConnectLayer, self).__init__()

    def forward(self, x, sub_layer):
        out = x
        out = sub_layer(out)
        out = out + x

        return out