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

        return x


class EGCN(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim=2000, n_gcn=2, n_hidden_gcn=64, activation=nn.ReLU,
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

        return x

class MLP(th.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden: List[int], dropout: float = 0.5):
        super().__init__()
        self.dropout = th.nn.Dropout(p=dropout)
        assert hidden
        ls = [th.nn.Linear(in_channels, hidden[0])]
        if len(hidden) > 1:
            ls += [th.nn.Linear(hd1, hd2) for hd1, hd2 in zip(hidden, hidden[1:])]
        ls += [th.nn.Linear(hidden[-1], out_channels)]
        self.layers = th.nn.ModuleList(ls)
        self.act = th.nn.SELU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
                x = self.dropout(x)

        return x


