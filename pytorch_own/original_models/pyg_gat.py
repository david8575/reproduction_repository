from torch_geometric.nn.models import GAT

def get_pyg_gat_model(in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
    return GAT(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=2,
        out_channels=out_channels,
        heads=heads,
        dropout=dropout
    )