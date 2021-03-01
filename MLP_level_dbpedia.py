from typing import List

import numpy as np
import torch as th
import pandas as pd
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from textgcn.lib.models import MLP

"""
TODO
"""


def csr_to_torch(csr):
    acoo = csr.tocoo()
    return th.sparse_coo_tensor(th.LongTensor([acoo.row.tolist(), acoo.col.tolist()]),
                                th.FloatTensor(acoo.data.astype(np.float32)),
                                size=csr.shape)


def append_feats(feats, top_labels):
    feats = feats.cpu()
    if isinstance(top_labels, np.ndarray):
        top_labels = th.from_numpy(top_labels)
    indices = th.nonzero(top_labels).t()
    values = top_labels[indices[0], indices[1]]
    if isinstance(top_labels, np.ndarray):
        values = th.from_numpy(values)
    mat = th.sparse.FloatTensor(indices, values, top_labels.shape)
    mat = mat.cpu()
    return th.cat([feats, mat], dim=1)



CPU_ONLY = False
EARLY_STOPPING = False
epochs = 500
lr = 0.05
dropout = 0.7
seed = 42
result_file = "results_dbpedia.csv"
model = MLP
np.random.seed(seed)
th.random.manual_seed(seed)
save_results = True
labels = "l3"


try:
    df = pd.read_csv(result_file)
except:
    df = pd.DataFrame(columns=["seed", "model", "hierarchy", "f1-macro", "accuracy"])

train = pd.read_csv("data/dbpedia/DBPEDIA_train.csv")
val = pd.read_csv("data/dbpedia/DBPEDIA_val.csv")
test = pd.read_csv("data/dbpedia/DBPEDIA_test.csv")

# save_path = None
raw_train = train['text']
y_train = train[labels]
y_top1_train = train["l1"]
y_top2_train = train["l2"]

# Train/val split
raw_val = val["text"]

y_val = val[labels]
y_top1_val = val["l1"]
y_top2_val = val["l2"]

tfidf = TfidfVectorizer(stop_words='english', max_df=0.9)
x_train = tfidf.fit_transform(raw_train)
y_train = y_train.tolist()
x_val = tfidf.transform(raw_val.tolist())


raw_test = test['text'].tolist()
y_test = test[labels].tolist()

x_test = tfidf.transform(raw_test)

le1 = LabelEncoder()
y_top1_train = le1.fit_transform(y_top1_train)
y_top1_val = le1.transform(y_top1_val)

le2 = LabelEncoder()
y_top2_train = le2.fit_transform(y_top2_train)
y_top2_val = le2.transform(y_top2_val)

le3 = LabelEncoder()
y_train = le3.fit_transform(y_train)
y_val = le3.transform(y_val)

print("Data loaded!")

del raw_train
del raw_val
del raw_test

x_train, x_val, x_test = map(csr_to_torch, [x_train, x_val, x_test])
y_train, y_val, y_test, y_top1_train, y_top1_val, y_top2_train, y_top2_val = \
    map(th.from_numpy, [y_train, y_val, y_test, y_top1_train, y_top1_val, y_top2_train, y_top2_val])

model1 = MLP(x_train.shape[1], len(np.unique(y_top1_train)), [256, 128], dropout=dropout)

criterion = th.nn.CrossEntropyLoss(reduction='mean')

device = th.device('cuda' if th.cuda.is_available() and not CPU_ONLY else 'cpu')
model1 = model1.to(device).float()
x_train = x_train.to(device)
y_top1_train = y_top1_train.to(device)
x_val = x_val.to(device)

print(f"x_val shape: {x_val.shape}")
print(f"x_train shape: {x_train.shape}")

# optimizer needs to be constructed AFTER the model was moved to GPU
optimizer = th.optim.Adam(model1.parameters(), lr=lr)
length = len(str(epochs))
print("### Training start (Top-Level)! ###")
for epoch in range(epochs):
    model1.train()
    outputs = model1(x_train)
    loss = criterion(outputs, y_top1_train)
    # performance tip: try set_to_none=True
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    model1.eval()
    with th.no_grad():
        logits = model1(x_train)
        logits_val = model1(x_val)
        pred_val = np.argmax(logits_val.cpu().numpy(), axis=1)
        pred_train = np.argmax(logits.cpu().numpy(), axis=1)
        f1_val = f1_score(y_top1_val.cpu(), pred_val, average='macro')
        acc_train = accuracy_score(y_top1_train.cpu(), pred_train)
        print(f"[{epoch + 1:{length}}] loss: {loss.item(): .3f}, "
              f"training accuracy: {acc_train: .3f}, val_f1: {f1_val: .3f}")


oh = OneHotEncoder(sparse=False)
train_hierarchy_1 = oh.fit_transform(y_top1_train.cpu().numpy().reshape(-1, 1))
val_hierarchy_1 = oh.fit_transform(y_top1_val.cpu().numpy().reshape(-1, 1))

x_train_1 = append_feats(x_train, train_hierarchy_1)
x_val_1 = append_feats(x_val, val_hierarchy_1)

"""
below is not adapted
"""

model2 = MLP(x_train_1.shape[1], len(np.unique(y_top2_train)), [256, 128], dropout=dropout)

criterion = th.nn.CrossEntropyLoss(reduction='mean')

device = th.device('cuda' if th.cuda.is_available() and not CPU_ONLY else 'cpu')
model2 = model2.to(device).float()
x_train = x_train.to(device).float()
y_train = y_train.to(device)
x_val = x_val.to(device).float()

print(f"x_val shape: {x_val.shape}")
print(f"x_train shape: {x_train.shape}")

# optimizer needs to be constructed AFTER the model was moved to GPU
optimizer = th.optim.Adam(model2.parameters(), lr=lr)
length = len(str(epochs))
print("### Training start (Top-Level)! ###")
for epoch in range(epochs):
    model2.train()
    outputs = model2(x_train)
    loss = criterion(outputs, y_train)
    # performance tip: try set_to_none=True
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    model2.eval()
    with th.no_grad():
        logits = model2(x_train)
        logits_val = model2(x_val)
        pred_val = np.argmax(logits_val.cpu().numpy(), axis=1)
        pred_train = np.argmax(logits.cpu().numpy(), axis=1)
        f1_val = f1_score(y_val.cpu(), pred_val, average='macro')
        acc_train = accuracy_score(y_train.cpu(), pred_train)
        print(f"[{epoch + 1:{length}}] loss: {loss.item(): .3f}, "
              f"training accuracy: {acc_train: .3f}, val_f1: {f1_val: .3f}")

print("Optimization finished!")

del x_train
del x_val
del y_train
del y_val

with th.no_grad():
    x_test = x_test.to(device)
    top_pred = th.nn.Softmax()(model1(x_test))
    x_test = append_feats(x_test, top_pred).to(device)
    pred_test = np.argmax(model2(x_test).cpu(), axis=1)
    acc_test = accuracy_score(y_test.cpu(), pred_test)
    f1 = f1_score(y_test.cpu().detach(), pred_test, average='macro')

print(f"Test Accuracy: {acc_test: .3f}")
print(f"F1-Macro: {f1: .3f}")


if save_results:
    i = df.index.max() + 1 if df.index.max() != np.nan else 0
    df.loc[i] = {'seed': seed, 'model': "MLP", 'hierarchy': "per-level", 'f1-macro': f1,
                 'accuracy': acc_test}
    df.to_csv(result_file)





