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
        if g.x.is_sparse:
            return self.forward_sparse(g)
        return self.forward_dense(g)

    def forward_dense(self, g):
        x = g.x
        for i, layer in enumerate(self.graph_layers):
            x = layer(x, g.edge_index, g.edge_attr)
            if i < len(self.graph_layers) - 1:
                x = self.activation(x)
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        return nn.Softmax(dim=-1)(x)

    def forward_sparse(self, g):
        # once again inspired by
        # https://kenqgu.com/classifying-asian-prejudice-in-tweets-during-covid-19-using-graph-convolutional-networks/
        x = g.x
        for i, layer in enumerate(self.graph_layers):
            # x = layer(x, g.edge_index, g.edge_attr)
            x = th.sparse.mm(x, layer.weight)
            if not layer.cached or layer.cached_result is None:
                edge_index, norm = layer.norm(g.edge_index, x.size(0), g.edge_attr, layer.improved, x.dtype)
                layer.cached_result = edge_index, norm
            else:
                edge_index, norm = self.cached_result
            x = layer.propagate(edge_index, x=x, norm=norm)
            if i < len(self.graph_layers) - 1:
                x = self.activation(x)
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        return nn.Softmax(dim=-1)(x)


