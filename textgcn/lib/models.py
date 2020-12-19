from torch import nn
from torch_geometric.nn import GCNConv


class MLP_GCN(nn.Module):
    def __init__(self, in_channels, out_channels, n_gcn = 2, n_mlp = 2, n_hidden_gcn = 64, n_hidden_mlp = 64,
                 activation = nn.SELU):
        super(MLP_GCN, self).__init__()
        self.activation = activation()
        self.graph_layers = nn.ParameterList([GCNConv(in_channels, n_hidden_gcn)])
        for i in range(n_gcn - 1):
            self.graph_layers.append(GCNConv(n_hidden_gcn, n_hidden_gcn))
        self.layers = nn.ParameterList([nn.Linear(n_hidden_gcn, n_hidden_mlp)])
        for i in range(n_mlp):
            self.layers.append(nn.Linear(n_hidden_mlp, n_hidden_mlp))
        self.layers.append(nn.Linear(n_hidden_mlp, out_channels))

    def forward(self, g):
        x = g.x
        for layer in self.graph_layers:
            x = layer(x, g.edge_index, g.edge_attrs)
            x = self.activation(x)
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x


class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, n_gcn = 2,n_hidden_gcn = 64, activation=nn.SELU):
        super(GCN, self).__init__()
        self.activation = activation()
        self.graph_layers = nn.ModuleList([GCNConv(in_channels, n_hidden_gcn)])
        for i in range(n_gcn - 2):
            self.layers.append(GCNConv(n_hidden_gcn, n_hidden_gcn))
        self.graph_layers.append(GCNConv(n_hidden_gcn, out_channels))

    def forward(self, g):
        x = g.x
        for i, layer in enumerate(self.graph_layers):
            x = layer(x, g.edge_index, g.edge_attr)
            if i < len(self.graph_layers) - 1:
                x = self.activation(x)

        return nn.Softmax(dim=-1)(x)
