import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch as th
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch_geometric import nn

from textgcn import Text2GraphTransformer
from textgcn.lib.models import *



CPU_ONLY = True
EARLY_STOPPING = False
epochs = 100
train_val_split = 0.1
lr = 0.05
save_model = False
dropout = 0.5
max_df = 0.4
seed = 44
result_file = "results_dbpedia.csv"
model = GCN
np.random.seed(seed)
th.random.manual_seed(seed)
save_results = True
labels = "l3"
window_size = 5
MAX_LENGTH = 15

try:
    df = pd.read_csv(result_file)
except:
    df = pd.DataFrame(columns=["seed", "model", "hierarchy", "f1-macro", "accuracy"])

train = pd.read_csv("data/dbpedia/DBPEDIA_train.csv")
val = pd.read_csv("data/dbpedia/DBPEDIA_val.csv")
test = pd.read_csv("data/dbpedia/DBPEDIA_test.csv")

save_path = None
# save_path = None
x = train['text'].tolist()
y = train[labels].tolist()

x_val = val['text'].tolist()
y_val = val[labels].tolist()

# Train/val split
val_idx = np.arange(len(x), len(x) + len(x_val))

x += x_val
y += y_val

x_test = test['text'].tolist()
y_test = test[labels].tolist()

test_idx = np.arange(len(x), len(x) + len(x_test))

x += x_test
y += y_test

y = LabelEncoder().fit_transform(y)
print("Data loaded!")

t2g = Text2GraphTransformer(n_jobs=8, min_df=100, save_path=None, verbose=1, max_df=max_df, window_size=window_size,
                        max_length=MAX_LENGTH)
# t2g = Text2GraphTransformer(n_jobs=8, min_df=1, save_path=save_path, verbose=1, max_df=1.0)
ls = os.listdir("textgcn/graphs")

g = t2g.fit_transform(x, y, test_idx=test_idx, val_idx=val_idx)
print("Graph built!")

# gcn = JumpingKnowledgeNetwork(g.x.shape[1], len(np.unique(y)), n_hidden_gcn=100, dropout=dropout, activation=th.nn.SELU)
# gcn = EGCN(g.x.shape[1], len(np.unique(y)), n_hidden_gcn=100, embedding_dim=2000, dropout=dropout)
gcn = model(g.x.shape[1], len(np.unique(y)), n_hidden_gcn=32, dropout=dropout)

criterion = th.nn.CrossEntropyLoss(reduction='mean')

device = th.device('cuda' if th.cuda.is_available() and not CPU_ONLY else 'cpu')
gcn = gcn.to(device).float()
g = g.to(device)

# optimizer needs to be constructed AFTER the model was moved to GPU
optimizer = th.optim.Adam(gcn.parameters(), lr=lr, amsgrad=True)
# optimizer = th.optim.SGD(gcn.parameters(), lr=0.05)
# optimizer = th.optim.Adadelta(gcn.parameters())
# scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, factor=0.5, patience=15, cooldown=5)
# scheduler = th.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.5, step_size_up=50)
history = []
length = len(str(epochs))
print(device)
print("#### TRAINING START ####")
time_start = datetime.now()
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
        val_loss = criterion(logits[g.val_mask], g.y[g.val_mask])
        pred_val = np.argmax(logits[g.val_mask].cpu().numpy(), axis=1)
        pred_train = np.argmax(logits[g.train_mask].cpu().numpy(), axis=1)
        f1_val = f1_score(g.y.cpu()[g.val_mask], pred_val, average='macro')
        acc_train = accuracy_score(g.y.cpu()[g.train_mask], pred_train)
        print(f"[{epoch + 1:{length}}] loss: {loss.item(): .3f}, "
              f"training accuracy: {acc_train: .3f}, val_f1: {f1_val: .3f}")
    history.append((loss.item(), f1_val))

    # scheduler.step(val_loss)

    if epoch > 5 and history[-5][0] < history[-1][0] and EARLY_STOPPING:
        print("early stopping activated!")
        break

print("Optimization finished!")
if save_model:
    print("Saving model...")
    th.save(gcn, f"models/gcn_{int(time.time())}.nn")

with th.no_grad():
    pred_test = np.argmax(gcn(g)[g.test_mask].cpu().detach().numpy(), axis=1)
    acc_test = accuracy_score(g.y.cpu()[g.test_mask].detach(), pred_test)
    f1 = f1_score(g.y.cpu()[g.test_mask].detach(), pred_test, average='macro')
    conf_mat = confusion_matrix(g.y.cpu()[g.test_mask].detach(), pred_test)

print(f"Test Accuracy: {acc_test: .3f}")
print(f"F1-Macro: {f1: .3f}")
print("Confusion matrix:")
print(conf_mat)

time_end = datetime.now()
print(f"Training took {time_end - time_start} for {epoch + 1} epochs.")

if save_results:
    i = df.index.max() + 1 if df.index.max() != np.nan else 0
    df.loc[i] = {'seed': seed, 'model': "GCN" if isinstance(gcn, GCN) else "EGCN", 'hierarchy': "flat", 'f1-macro': f1, 'accuracy': acc_test}
    df.to_csv(result_file, index=False)

