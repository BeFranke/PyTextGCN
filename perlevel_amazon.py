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

CPU_ONLY = False
EARLY_STOPPING = False
epochs = 500
train_val_split = 0.1
lr1 = 0.05
lr2 = 0.05
save_model = False
dropout1 = 0.7
dropout2 = 0.7
model = GCN
max_df1 = 0.6
max_df2 = 0.6

seed = 44
result_file = "results.csv"

np.random.seed(seed)
th.random.manual_seed(seed)

try:
    df = pd.read_csv(result_file)
except:
    df = pd.DataFrame(columns=["seed", "model", "hierarchy", "f1-macro", "accuracy"])


train = pd.read_csv("data/amazon/train.csv")
test = pd.read_csv("data/amazon/test.csv")

# save_path = "textgcn/graphs/"
save_path = None
x = train['Text'].tolist()
y = train['Cat2'].tolist()
y_top = train['Cat1'].tolist()

# Train/val split
val_idx = np.random.choice(len(x), int(train_val_split * len(x)), replace=False)
train_idx = np.array([x for x in range(len(x)) if x not in val_idx])

x_test = test['Text'].tolist()
y_test = test['Cat2'].tolist()
y_test_top = test['Cat1'].tolist()

test_idx = np.arange(len(x), len(x) + len(x_test))

y = y + y_test
y_top = y_top + y_test_top
x = x + x_test

y = LabelEncoder().fit_transform(y)
y_top = LabelEncoder().fit_transform(y_top)
print("Data loaded!")

t2g = Text2GraphTransformer(n_jobs=8, min_df=5, save_path=save_path, verbose=1, max_df=max_df1)


g1 = t2g.fit_transform(x, y_top, test_idx=test_idx, val_idx=val_idx, hierarchy_feats=None)

print("Graph built!")

gcn1 = model(g1.x.shape[1], len(np.unique(y_top)), n_hidden_gcn=100, dropout=dropout1)

criterion = th.nn.CrossEntropyLoss(reduction='mean')

device = th.device('cuda' if th.cuda.is_available() and not CPU_ONLY else 'cpu')
gcn1 = gcn1.to(device).float()
g1 = g1.to(device)

# optimizer needs to be constructed AFTER the model was moved to GPU
optimizer = th.optim.Adam(gcn1.parameters(), lr=lr1)

length = len(str(epochs))
print(device)
print("#### TRAINING START ####")
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
    hierarchy_true = th.nn.functional.softmax(gcn1(g1)[g1.n_vocab:]).cpu().numpy()

hierarchy = OneHotEncoder(sparse=False).fit_transform(y_top.reshape(-1, 1))
print(f"shape of hierarchy: {hierarchy.shape}")
print(f"shape of hierarchy_true: {hierarchy_true.shape}")

with open("textgcn/models/amazon/lvl1", "wb") as f:
    th.save(gcn1, f)

del gcn1
del g1
t2g = Text2GraphTransformer(n_jobs=8, min_df=5, save_path=save_path, verbose=1, max_df=max_df2)
g2 = t2g.fit_transform(x, y, test_idx=test_idx, val_idx=val_idx, hierarchy_feats=hierarchy)
gcn2 = model(g2.x.shape[1], len(np.unique(y)), n_hidden_gcn=100, dropout=dropout2)

gcn2 = gcn2.to(device)
g2 = g2.to(device)

# optimizer needs to be constructed AFTER the model was moved to GPU
optimizer = th.optim.Adam(gcn2.parameters(), lr=lr2)

length = len(str(epochs))
print(device)
print("#### TRAINING START ####")
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

assert hierarchy.shape == hierarchy_true.shape

g2 = t2g.fit_transform(x, y, test_idx=test_idx, val_idx=val_idx, hierarchy_feats=hierarchy_true)
g2 = g2.to(device)

with th.no_grad():
    pred_test = np.argmax(gcn2(g2)[g2.test_mask].cpu().detach().numpy(), axis=1)
    acc_test = accuracy_score(g2.y.cpu()[g2.test_mask].detach(), pred_test)
    f1 = f1_score(g2.y.cpu()[g2.test_mask].detach(), pred_test, average='macro')
    conf_mat = confusion_matrix(g2.y.cpu()[g2.test_mask].detach(), pred_test)

print(f"Test Accuracy: {acc_test: .3f}")
print(f"F1-Macro: {f1: .3f}")
print("Confusion matrix:")
print(conf_mat)

time_end = datetime.now()
print(f"Training took {time_end - time_start} for {epoch + 1} epochs.")

i = df.index.max() + 1 if df.index.max() != np.nan else 0
df.loc[i] = {'seed': seed, 'model': "GCN" if isinstance(gcn2, GCN) else "EGCN", 'hierarchy': "per-level", 'f1-macro': f1, 'accuracy': acc_test}
df.to_csv(result_file, index=False)
