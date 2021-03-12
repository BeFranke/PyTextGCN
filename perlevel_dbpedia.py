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
import torch as th
from textgcn import Text2GraphTransformer
from textgcn.lib.models import JumpingKnowledgeNetwork, GCN, EGCN

CPU_ONLY = True
EARLY_STOPPING = False
epochs = 100
lr = 0.05
save_model = False
dropout = 0.5
model = GCN
max_df = 0.4
n_hidden = 32
seed = 44
result_file = "results_dbpedia.csv"
window_size = 5
MAX_LENGTH = 15
save_results = True
np.random.seed(seed)
th.random.manual_seed(seed)

try:
    df = pd.read_csv(result_file)
except:
    df = pd.DataFrame(columns=["seed", "model", "hierarchy", "f1-macro", "accuracy"])


train = pd.read_csv("data/dbpedia/DBPEDIA_train.csv")
val = pd.read_csv("data/dbpedia/DBPEDIA_val.csv")
test = pd.read_csv("data/dbpedia/DBPEDIA_test.csv")

# save_path = "textgcn/graphs/"
save_path = None
x = train['text'].tolist()
y = train['l3'].tolist()
y_top1 = train['l1'].tolist()
y_top2 = train['l2'].tolist()

x_val = val['text'].tolist()
y_val = val['l3'].tolist()
y_top1_val = val['l1'].tolist()
y_top2_val = val['l2'].tolist()

x_test = test['text'].tolist()
y_test = test['l3'].tolist()
y_top1_test = test['l1'].tolist()
y_top2_test = test['l2'].tolist()

val_idx = np.arange(len(x), len(x) + len(x_val))

x += x_val
y += y_val
y_top1 += y_top1_val
y_top2 += y_top2_val

test_idx = np.arange(len(x), len(x) + len(x_test))

x += x_test
y += y_test
y_top1 += y_top1_test
y_top2 += y_top2_test

y = LabelEncoder().fit_transform(y)
y_top1 = LabelEncoder().fit_transform(y_top1)
y_top2 = LabelEncoder().fit_transform(y_top2)

del x_val
del x_test
del y_top1_val
del y_top2_val
del y_top1_test
del y_top2_test

print("Data loaded!")

t2g = Text2GraphTransformer(n_jobs=1, min_df=100, save_path=save_path, verbose=1, max_df=max_df, max_length=MAX_LENGTH,
                            window_size=window_size)


g1 = t2g.fit_transform(x, y_top1, test_idx=test_idx, val_idx=val_idx, hierarchy_feats=None)

print("Graph built!")

gcn1 = model(g1.x.shape[1], len(np.unique(y_top1)), n_hidden_gcn=n_hidden, dropout=dropout)

criterion = th.nn.CrossEntropyLoss(reduction='mean')

device = th.device('cuda' if th.cuda.is_available() and not CPU_ONLY else 'cpu')
gcn1 = gcn1.to(device).float()
g1 = g1.to(device)

# optimizer needs to be constructed AFTER the model was moved to GPU
optimizer = th.optim.Adam(gcn1.parameters(), lr=lr)

length = len(str(epochs))
print(device)
print("#### TRAINING START (Level 1)####")
time_start = datetime.now()
for epoch in range(epochs):
    gcn1.train()
    outputs = gcn1(g1)[g1.train_mask]
    loss = criterion(outputs, g1.y[g1.train_mask])
    # performance tip: try set_to_none=True
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    gcn1.eval()
    with th.no_grad():
        logits = gcn1(g1)
        val_loss = criterion(logits[g1.val_mask], g1.y[g1.val_mask])
        pred_val = np.argmax(logits[g1.val_mask].cpu().numpy(), axis=1)
        pred_train = np.argmax(logits[g1.train_mask].cpu().numpy(), axis=1)
        acc_val = accuracy_score(g1.y.cpu()[g1.val_mask], pred_val)
        acc_train = accuracy_score(g1.y.cpu()[g1.train_mask], pred_train)
        print(f"[{epoch + 1:{length}}] loss: {loss.item(): .3f}, "
              f"training accuracy: {acc_train: .3f}, val_accuracy: {acc_val: .3f}")

with th.no_grad():
    hierarchy_true1 = th.nn.functional.softmax(gcn1(g1)[g1.n_vocab:]).cpu().numpy()

hierarchy1 = OneHotEncoder(sparse=False).fit_transform(y_top1.reshape(-1, 1))
print(f"shape of hierarchy: {hierarchy1.shape}")
print(f"shape of hierarchy_true: {hierarchy_true1.shape}")

del gcn1
del g1


g2 = t2g.fit_transform(x, y_top2, test_idx=test_idx, val_idx=val_idx, hierarchy_feats=hierarchy1)
gcn2 = model(g2.x.shape[1], len(np.unique(y_top2)), n_hidden_gcn=n_hidden, dropout=dropout)

gcn2 = gcn2.to(device)
g2 = g2.to(device)

# optimizer needs to be constructed AFTER the model was moved to GPU
optimizer = th.optim.Adam(gcn2.parameters(), lr=lr)

length = len(str(epochs))
print(device)
print("#### TRAINING START (Level 2)####")
time_start = datetime.now()
for epoch in range(epochs):
    gcn2.train()
    outputs = gcn2(g2)[g2.train_mask]
    loss = criterion(outputs, g2.y[g2.train_mask])
    # performance tip: try set_to_none=True
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    gcn2.eval()
    with th.no_grad():
        logits = gcn2(g2)
        val_loss = criterion(logits[g2.val_mask], g2.y[g2.val_mask])
        pred_val = np.argmax(logits[g2.val_mask].cpu().numpy(), axis=1)
        pred_train = np.argmax(logits[g2.train_mask].cpu().numpy(), axis=1)
        acc_val = accuracy_score(g2.y.cpu()[g2.val_mask], pred_val)
        acc_train = accuracy_score(g2.y.cpu()[g2.train_mask], pred_train)
        print(f"[{epoch + 1:{length}}] loss: {loss.item(): .3f}, "
              f"training accuracy: {acc_train: .3f}, val_accuracy: {acc_val: .3f}")

assert hierarchy1.shape == hierarchy_true1.shape

with th.no_grad():
    hierarchy_true2 = th.nn.functional.softmax(gcn2(g2)[g2.n_vocab:]).cpu().numpy()

del gcn2
del g2

hierarchy2 = OneHotEncoder(sparse=False).fit_transform(y_top2.reshape(-1, 1))
print(f"shape of hierarchy: {hierarchy2.shape}")
print(f"shape of hierarchy_true: {hierarchy_true2.shape}")


g3 = t2g.fit_transform(x, y, test_idx=test_idx, val_idx=val_idx, hierarchy_feats=hierarchy2)
gcn3 = model(g3.x.shape[1], len(np.unique(y)), n_hidden_gcn=n_hidden, dropout=dropout)

gcn3 = gcn3.to(device)
g3 = g3.to(device)

# optimizer needs to be constructed AFTER the model was moved to GPU
optimizer = th.optim.Adam(gcn3.parameters(), lr=lr)

length = len(str(epochs))
print(device)
print("#### TRAINING START (Level 3)####")
time_start = datetime.now()
for epoch in range(epochs):
    gcn3.train()
    outputs = gcn3(g3)[g3.train_mask]
    loss = criterion(outputs, g3.y[g3.train_mask])
    # performance tip: try set_to_none=True
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    gcn3.eval()
    with th.no_grad():
        logits = gcn3(g3)
        val_loss = criterion(logits[g3.val_mask], g3.y[g3.val_mask])
        pred_val = np.argmax(logits[g3.val_mask].cpu().numpy(), axis=1)
        pred_train = np.argmax(logits[g3.train_mask].cpu().numpy(), axis=1)
        acc_val = accuracy_score(g3.y.cpu()[g3.val_mask], pred_val)
        acc_train = accuracy_score(g3.y.cpu()[g3.train_mask], pred_train)
        print(f"[{epoch + 1:{length}}] loss: {loss.item(): .3f}, "
              f"training accuracy: {acc_train: .3f}, val_accuracy: {acc_val: .3f}")

assert hierarchy2.shape == hierarchy_true2.shape

g3 = t2g.fit_transform(x, y, test_idx=test_idx, val_idx=val_idx, hierarchy_feats=hierarchy_true2)
g3 = g3.to(device)

with th.no_grad():
    pred_test = np.argmax(gcn3(g3)[g3.test_mask].cpu().detach().numpy(), axis=1)
    acc_test = accuracy_score(g3.y.cpu()[g3.test_mask].detach(), pred_test)
    f1 = f1_score(g3.y.cpu()[g3.test_mask].detach(), pred_test, average='macro')
    conf_mat = confusion_matrix(g3.y.cpu()[g3.test_mask].detach(), pred_test)

print(f"Test Accuracy: {acc_test: .3f}")
print(f"F1-Macro: {f1: .3f}")
print("Confusion matrix:")
print(conf_mat)

time_end = datetime.now()
print(f"Training took {time_end - time_start} for {epoch + 1} epochs.")

if save_results:
    i = df.index.max() + 1 if df.index.max() != np.nan else 0
    df.loc[i] = {'seed': seed, 'model': "GCN", 'hierarchy': "per-level", 'f1-macro': f1, 'accuracy': acc_test}
    df.to_csv(result_file, index=False)
