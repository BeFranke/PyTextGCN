from torch import nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, n_gcn = 2, n_mlp = 2, n_hidden_gcn = 64, n_hidden_mlp = 64):
        super(GCN, self).__init__()
        self.layers = [GCNConv(in_channels, n_hidden_gcn)]
        self.layers.append(nn.SELU())
        for i in range(n_gcn - 1):
            self.layers.append(GCNConv(n_hidden_gcn, n_hidden_gcn))
            self.layers.append(nn.SELU())
        self.layers.append(nn.Linear(n_hidden_gcn, n_hidden_mlp))
        self.layers.append(nn.SELU())
        for i in range(n_mlp):
            self.layers.append(nn.Linear(n_hidden_mlp, n_hidden_mlp))
            self.layers.append(nn.SELU())
        self.layers.append(nn.Linear(n_hidden_mlp, out_channels))
        self.layers.append(nn.Softmax(dim=-1))

    def forward(self, g):
        x = g.x
        for layer in self.layers:
            x = layer(x, g.edge_index, g.edge_attrs)
        return x
