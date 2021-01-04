from torch import nn
import torch as th
from torch_geometric.nn import GCNConv, GraphConv


class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, n_gcn=2, n_hidden_gcn=64, activation=nn.ReLU, dropout=0.5):
        super(GCN, self).__init__()
        self.activation = activation()
        self.dropout = dropout
        self.layers = nn.ModuleList([GCNConv(in_channels, n_hidden_gcn, add_self_loops=True)])
        for i in range(n_gcn - 2):
            self.layers.append(GCNConv(n_hidden_gcn, n_hidden_gcn))

        self.layers.append(GCNConv(n_hidden_gcn, out_channels))

    def forward(self, g):
        x = g.x
        for i, layer in enumerate(self.layers):
            x = layer(x, g.edge_index, g.edge_attr)
            if i < len(self.layers) - 1:
                # x = self.activation(x)    # GCN includes RELU
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        return nn.Softmax(dim=-1)(x)


class JumpingKnowledge(nn.Module):
    # currently just a GCN with skip connections
    def __init__(self, in_channels, out_channels, n_gcn=2, n_hidden_gcn=64, activation=nn.ReLU, dropout=0.5):
        super().__init__()
        self.activation = activation()
        self.dropout = dropout
        self.layers = nn.ModuleList([GCNConv(in_channels, n_hidden_gcn, add_self_loops=True)])
        for i in range(n_gcn - 2):
            self.layers.append(GCNConv(n_hidden_gcn, n_hidden_gcn))
        self.layers.append(GCNConv(n_hidden_gcn, n_hidden_gcn))
        self.layers.append(th.nn.Linear(n_hidden_gcn * n_gcn, out_channels))

    def forward(self, g):
        x = g.x
        acts = []
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = layer(x, g.edge_index, g.edge_attr)
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)
                acts += [x]
            else:
                x = layer(th.cat(acts, dim=-1))

        return nn.Softmax(dim=-1)(x)


class HierarchyGNN(nn.Module):
    """
    Basic idea: GNN for heterogeneous Graph, then throw away word-nodes and compute MLP on doc-nodes with optional
    input from higher hierarchies
    """
    def __init__(self, in_feats, n_classes, n_hidden=64, n_layers=2, dropout=0.5, activation=nn.ReLU,
                 mlp_hidden=64, mlp_layers=2, hierarchy_feat_dim=0, graph_layer=GCNConv, residual=False):
        super().__init__()
        self.residual = residual
        assert n_layers > 1
        self.activation = activation()
        self.dropout = dropout
        self.layers_gcn = nn.ModuleList([graph_layer(in_channels=in_feats, out_channels=n_hidden)])
        for _ in range(n_layers - 1):
            self.layers_gcn.append(graph_layer(n_hidden, n_hidden))
        # between these layers, word-nodes are discarded
        if mlp_layers > 1:
            self.layers_mlp = nn.ModuleList([th.nn.Linear(n_hidden + hierarchy_feat_dim, mlp_hidden)])
            for _ in range(mlp_layers - 2):
                self.layers_mlp.append(th.nn.Linear(mlp_hidden, mlp_hidden))
            self.layers_mlp.append(th.nn.Linear(mlp_hidden, n_classes))
        else:
            self.layers_mlp = nn.ModuleList([th.nn.Linear(n_hidden + hierarchy_feat_dim, n_classes)])

    def forward(self, g, hierachy_feat=None):
        activations = []
        x = g.x
        # regular GNN feedforward
        for layer in self.layers_gcn:
            x = layer(x=x,
                      edge_index=g.edge_index,
                      edge_weight=g.edge_attr)
            x = self.activation(x)
            x = th.nn.functional.dropout(x, p=self.dropout, training=self.training)
            activations.append(x)

        # add hierarchy information if present
        if hierachy_feat is not None:
            doc_mask = th.logical_or(th.logical_or(g.val_mask, g.test_mask), g.train_mask)
            if len(hierachy_feat.shape < 2):
                hierachy_feat = hierachy_feat[:, None]
            hf = th.zeros(x.shape([1]))
            hf[doc_mask] = hierachy_feat
            x = th.hstack([x, hierachy_feat])

        # MLP feedforward
        for i, layer in enumerate(self.layers_mlp):
            if i == len(self.layers_mlp) - 1 and self.residual:
                x = th.cat(activations, dim=-1)
            x = layer(x)
            if i < len(self.layers_mlp) - 1:
                x = self.activation(x)
                x = th.nn.functional.dropout(x, p=self.dropout, training=self.training)
                activations.append(x)

        return nn.Softmax(dim=-1)(x)
