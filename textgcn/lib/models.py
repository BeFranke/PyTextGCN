from torch import nn
import torch as th
from torch_geometric.nn import GCNConv, GraphConv, JumpingKnowledge, GENConv, DeepGCNLayer


class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, n_gcn=2, n_hidden_gcn=64, activation=nn.ReLU, dropout=0.5):
        super(GCN, self).__init__()
        self.activation = activation()
        self.dropout = dropout
        self.layers = nn.ModuleList([GCNConv(in_channels, n_hidden_gcn, add_self_loops=True)])
        for i in range(n_gcn - 2):
            self.layers.append(GCNConv(n_hidden_gcn, n_hidden_gcn, add_self_loops=True))

        self.layers.append(GCNConv(n_hidden_gcn, out_channels, add_self_loops=True))

    def forward(self, g):
        x = g.x
        for i, layer in enumerate(self.layers):
            x = layer(x, g.edge_index, g.edge_attr)
            if i < len(self.layers) - 1:
                # x = self.activation(x)    # GCN includes RELU
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        return nn.Softmax(dim=-1)(x)

class EGCN(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim=1000, n_gcn=2, n_hidden_gcn=64, activation=nn.ReLU,
                 dropout=0.5):
        super().__init__()
        self.activation = activation()
        self.dropout = dropout
        self.layers = nn.ModuleList([nn.Linear(in_channels, embedding_dim),
                                     GCNConv(embedding_dim, n_hidden_gcn, add_self_loops=True)])
        for i in range(n_gcn - 2):
            self.layers.append(GCNConv(n_hidden_gcn, n_hidden_gcn, add_self_loops=True))

        self.layers.append(GCNConv(n_hidden_gcn, out_channels, add_self_loops=True))

    def forward(self, g):
        x = g.x
        x = self.layers[0](x)
        x = nn.SELU()(x)
        for i, layer in enumerate(self.layers[1:]):
            x = layer(x, g.edge_index, g.edge_attr)
            if i < len(self.layers) - 1:
                # x = self.activation(x)    # GCN includes RELU
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        return nn.Softmax(dim=-1)(x)



class JumpingKnowledgeNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, n_gcn=2, n_hidden_gcn=64, activation=nn.ReLU, dropout=0.5):
        super().__init__()
        self.activation = activation()
        self.dropout = dropout
        self.layers = nn.ModuleList([GCNConv(in_channels, n_hidden_gcn, add_self_loops=True)])
        for i in range(n_gcn - 2):
            self.layers.append(GCNConv(n_hidden_gcn, n_hidden_gcn, add_self_loops=True))
        self.layers.append(GCNConv(n_hidden_gcn, n_hidden_gcn, add_self_loops=True))
        self.jk = JumpingKnowledge(mode="lstm", channels=n_hidden_gcn, num_layers=n_gcn)
        self.lin = nn.Linear(n_hidden_gcn, out_channels)

    def forward(self, g):
        x = g.x
        acts = []
        for i, layer in enumerate(self.layers):
            x = layer(x, g.edge_index, g.edge_attr)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
            acts += [x]

        x = self.jk(acts)
        x = self.activation(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        x = self.lin(x)

        return nn.Softmax(dim=-1)(x)


class JumpingGCN(nn.Module):
    """A 2-layer GCN with skip connections into another GCN. Pretty wacky."""
    def __init__(self, in_channels, out_channels, n_hidden_gcn=64, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.l1 = GCNConv(in_channels, n_hidden_gcn)
        self.l2 = GCNConv(n_hidden_gcn, n_hidden_gcn)
        self.l3 = GCNConv(n_hidden_gcn * 2, out_channels)

    def forward(self, g):
        x = g.x
        h1 = self.l1(x, g.edge_index, g.edge_attr)
        h1 = nn.functional.dropout(h1, p=self.dropout, training=self.training)
        h2 = self.l2(h1, g.edge_index, g.edge_attr)
        h2 = nn.functional.dropout(h2, p=self.dropout, training=self.training)
        h3 = self.l3(th.cat([h1, h2], dim=-1), g.edge_index, g.edge_attr)
        return nn.Softmax(dim=-1)(h3)

