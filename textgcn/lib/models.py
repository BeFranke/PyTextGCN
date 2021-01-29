from torch import nn
import torch as th
from torch_geometric.nn import GCNConv, GraphConv, JumpingKnowledge, GENConv, DeepGCNLayer, GATConv, SGConv


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


class SGAT(nn.Module):
    def __init__(self, in_channels, out_channels, K=2, n_hidden=64, dropout=0.5, heads=2, activation=nn.SELU):
        super().__init__()
        self.dropout = dropout
        self.activation = activation()
        self.sg = SGConv(in_channels, n_hidden, K)
        self.gat = GATConv(n_hidden, out_channels, heads, concat=False)

    def forward(self, g):
        x = g.x
        x = self.sg(x, g.edge_index, g.edge_attr)
        x = self.activation(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.gat(x, g.edge_index)
        return x


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
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        for i, layer in enumerate(self.layers[1:]):
            x = layer(x, g.edge_index, g.edge_attr)
            if i < len(self.layers) - 1:
                # x = self.activation(x)    # GCN includes RELU
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        return nn.Softmax(dim=-1)(x)


class EGCAN(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim=1000, n_gcn=2, n_hidden_gcn=64, activation=nn.ReLU,
                 dropout=0.5):
        super().__init__()
        self.activation = activation()
        self.dropout = dropout
        self.layers = nn.ModuleList([nn.Linear(in_channels, embedding_dim),
                                     GCNConv(embedding_dim, n_hidden_gcn, add_self_loops=True)])
        for i in range(n_gcn - 1):
            self.layers.append(GCNConv(n_hidden_gcn, n_hidden_gcn, add_self_loops=True))

        self.layers.append(GATConv(n_hidden_gcn, out_channels, heads=2, concat=False))

    def forward(self, g):
        x = g.x
        x = self.layers[0](x)
        x = nn.SELU()(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        for i, layer in enumerate(self.layers[1:]):
            if i < len(self.layers) - 2:
                # x = self.activation(x)    # GCN includes RELU
                x = layer(x, g.edge_index, g.edge_attr)
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)
            else:
                x = layer(x, g.edge_index)

        # return nn.Softmax(dim=-1)(x)
        return x


class GCAN(nn.Module):
    def __init__(self, in_channels, out_channels, n_gcn=2, n_hidden_gcn=64, activation=nn.ReLU,
                 dropout=0.5):
        super().__init__()
        self.activation = activation()
        self.dropout = dropout
        self.layers = nn.ModuleList([GCNConv(in_channels, n_hidden_gcn, add_self_loops=True)])
        for i in range(n_gcn):
            self.layers.append(GCNConv(n_hidden_gcn, n_hidden_gcn, add_self_loops=True))

        self.layers.append(GATConv(n_hidden_gcn, out_channels, heads=1, concat=False))

    def forward(self, g):
        x = g.x
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                # x = self.activation(x)    # GCN includes RELU
                x = layer(x, g.edge_index, g.edge_attr)
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)
            else:
                x = layer(x, g.edge_index)

        # return nn.Softmax(dim=-1)(x)
        return x


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

