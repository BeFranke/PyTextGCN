from torch import nn
import torch as th
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, n_gcn=2, n_hidden_gcn=64, activation=nn.ReLU, dropout=0.5):
        super(GCN, self).__init__()
        self.activation = activation()
        self.dropout = dropout
        self.graph_layers = nn.ModuleList([GCNConv(in_channels, n_hidden_gcn, add_self_loops=True)])
        for i in range(n_gcn - 2):
            self.layers.append(GCNConv(n_hidden_gcn, n_hidden_gcn))
        self.graph_layers.append(GCNConv(n_hidden_gcn, out_channels))

    def forward(self, g):
        x = g.x
        for i, layer in enumerate(self.graph_layers):
            x = layer(x, g.edge_index, g.edge_attr)
            if i < len(self.graph_layers) - 1:
                x = self.activation(x)
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        return nn.Softmax(dim=-1)(x)
