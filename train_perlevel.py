import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch as th
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch_geometric import nn

from textgcn import Text2GraphTransformer
from textgcn.lib.models import JumpingKnowledgeNetwork, GCN


CPU_ONLY = False
EARLY_STOPPING = False
epochs = 200
train_val_split = 0.15
lr = 0.3
save_model = False
LEVELS = 2
dataset_name = "amazon"
max_df = 0.9
min_df = 5
n_hidden = 64

# load dataset
train = pd.read_csv(f"data/{dataset_name}/train.csv")


def train_level(graph, lvl, lr, dset_name):
    print(f"Training Level {lvl}")
    gcn = GCN(graph.x.shape[1], len(np.unique(graph.y)), n_hidden_gcn=n_hidden, n_gcn=2)

    criterion = th.nn.CrossEntropyLoss(reduction='mean')

    device = th.device('cuda' if th.cuda.is_available() and not CPU_ONLY else 'cpu')
    gcn = gcn.to(device).float()
    g = graph.to(device)

    # optimizer needs to be constructed AFTER the model was moved to GPU
    optimizer = th.optim.Adam(gcn.parameters(), lr=lr)
    length = len(str(epochs))
    print(device)
    print("#### TRAINING START ####")
    for epoch in range(epochs):
        gcn.train()
        outputs = gcn(g)[g.train_mask]
        loss = criterion(outputs, g.y[g.train_mask])
        # performance tip: try set_to_none=True
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        gcn.eval()
        with th.no_grad():
            logits = gcn(g)
            pred_train = np.argmax(logits[g.train_mask].cpu().numpy(), axis=1)
            acc_train = accuracy_score(g.y.cpu()[g.train_mask], pred_train)
            print(f"[{epoch + 1:{length}}] loss: {loss.item(): .3f}, "
                  f"training accuracy: {acc_train: .3f}")

    print("Optimization finished!")
    print("Saving model...")
    th.save(gcn, f"models/gcn_{dset_name}_lvl{lvl}_perlevel.nn")

# save_path = "textgcn/graphs/"
save_path = None
x = train['Text'].tolist()
y_2 = train['Cat2'].tolist()
y_1 = train['Cat1'].tolist()

val_idx = []
train_idx = np.array([x for x in range(len(x)) if x not in val_idx])

y_1 = LabelEncoder().fit_transform(y_1)
y_2 = LabelEncoder().fit_transform(y_2)
print("Data loaded!")

t2g = Text2GraphTransformer(n_jobs=8, min_df=min_df, save_path=save_path, verbose=1, max_df=max_df)
hierarchy = OneHotEncoder(sparse=False).fit_transform(y_1.reshape(-1, 1))


g_2 = t2g.fit_transform(x, y_2, test_idx=val_idx, val_idx=None, hierarchy_feats=hierarchy)
g_1 = t2g.fit_transform(x, y_1, test_idx=val_idx, val_idx=None, hierarchy_feats=None)
graphs = [g_1, g_2]
print("Graphs built!")

for i, graph in enumerate(graphs):
    train_level(graph, i, lr, dataset_name)
