from torch import nn
import torch as th
from torch_geometric.nn import GCNConv, JumpingKnowledge, GATConv


class GCN(nn.Module):
    # This GCN implements the original Kipf & Welling version and is therefore not suited for inductive training
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


class HierarchyGAT(nn.Module):
    """
    Basic idea: GAT for heterogeneous Graph, then throw away word-nodes and compute MLP on doc-nodes with optional
    input from higher hierarchies
    """
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, doc_map, heads=1, dropout=0.5, activation=nn.ReLU,
                 mlp_hidden=64, mlp_layers=2):
        super().__init__()
        self.doc_map = doc_map
        assert n_layers > 1
        self.activation = activation()
        self.dropout = dropout
        self.layers_gcn = nn.ModuleList([GATConv(in_feats, n_hidden, heads, dropout=dropout)])
        for _ in range(n_layers - 1):
            self.layers_gcn.append(GATConv(n_hidden, n_hidden, heads, dropout=dropout))
        # between these layers, word-nodes are discarded
        self.layers_mlp = nn.ModuleList([th.nn.Linear(th.sum(doc_map).item(), mlp_hidden)])
        for _ in range(mlp_layers - 2):
            self.layers_mlp.append(th.nn.Linear(mlp_hidden, mlp_hidden))
        self.layers_mlp.append(th.nn.Linear(mlp_hidden, n_classes))

    def forward(self, g, hierachy_feat=None):
        x = g.x
        # regular GNN feedforward
        for layer in self.layers_gcn:
            x = layer(x, g.edge_index, g.edge_attr)
            x = self.activation(x)

        # throw away word-nodes
        x = x[self.doc_map]

        # add hierarchy information if present
        if hierachy_feat is not None:
            x = th.hstack([x, hierachy_feat.unsqueeze(1)])

        # MLP feedforward
        for i, layer in self.layers_mlp:
            x = layer(x)
            if i < len(self.layers_mlp) - 1:
                x = self.activation(x)
                x = th.nn.functional.dropout(x, p=self.dropout, training=self.training)

            return nn.Softmax(dim=-1)(x)
