from typing import List

import numpy as np
import torch as th
import pandas as pd
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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


CPU_ONLY = False
EARLY_STOPPING = False
epochs = 50
train_val_split = 0.1
lr = 0.05
dropout = 0.7
seed = 42
result_file = "results.csv"
model = MLP
np.random.seed(seed)
th.random.manual_seed(seed)
save_results = True
labels = "Cat2"


try:
    df = pd.read_csv(result_file)
except:
    df = pd.DataFrame(columns=["seed", "model", "hierarchy", "f1-macro", "accuracy"])

train = pd.read_csv("data/amazon/train.csv")
test = pd.read_csv("data/amazon/test.csv")

save_path = "textgcn/graphs/"
# save_path = None
raw = train['Text']
y_train = train[labels]
y_top_train = train["Cat1"]

# Train/val split
val_idx = np.random.choice(len(raw), int(train_val_split * len(raw)), replace=False)
val_mask = np.zeros(len(raw), dtype=np.bool)
val_mask[val_idx] = 1
raw_val = raw[val_mask]
raw_train = raw[np.logical_not(val_mask)]

y_val = y_train[val_mask]
y_top_val = y_top_train[val_mask]
y_train = y_train[np.logical_not(val_mask)]
y_top_train = y_top_train[np.logical_not(val_mask)]

tfidf = TfidfVectorizer(stop_words='english', max_df=0.9)
x_train = tfidf.fit_transform(raw_train)
y_train = y_train.tolist()
x_val = tfidf.transform(raw_val.tolist())


raw_test = test['Text'].tolist()
y_test = test[labels].tolist()
y_top_test = test["Cat1"].tolist()

x_test = tfidf.transform(raw_test)

le1 = LabelEncoder()
y_top_train = le1.fit_transform(y_top_train)
y_top_val = le1.transform(y_top_val)
y_top_test = le1.transform(y_top_test)

le2 = LabelEncoder()
y_train = le2.fit_transform(y_train)
y_val = le2.transform(y_val)
y_test = le2.transform(y_test)


print("Data loaded!")

del raw
del raw_train
del raw_val
del raw_test

x_train, x_val, x_test = map(csr_to_torch, [x_train, x_val, x_test])
y_train, y_val, y_test, y_top_train, y_top_val, y_to_test = \
    map(th.from_numpy, [y_train, y_val, y_test, y_top_train, y_top_val, y_top_test])

model1 = MLP(x_train.shape[1], len(np.unique(y_top_train)), [256, 128], dropout=dropout)

criterion = th.nn.CrossEntropyLoss(reduction='mean')

device = th.device('cuda' if th.cuda.is_available() and not CPU_ONLY else 'cpu')
model1 = model1.to(device).float()
x_train = x_train.to(device)
y_top_train = y_top_train.to(device)
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
    loss = criterion(outputs, y_top_train)
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
        f1_val = f1_score(y_top_val.cpu(), pred_val, average='macro')
        acc_train = accuracy_score(y_top_train.cpu(), pred_train)
        print(f"[{epoch + 1:{length}}] loss: {loss.item(): .3f}, "
              f"training accuracy: {acc_train: .3f}, val_f1: {f1_val: .3f}")

print("Optimization finished!")

oh = OneHotEncoder(sparse=False)
train_hierarchy = oh.fit_transform(y_top_train.cpu().numpy().reshape(-1, 1))
val_hierarchy = oh.fit_transform(y_top_val.cpu().numpy().reshape(-1, 1))

x_train = append_feats(x_train, train_hierarchy)
x_val = append_feats(x_val, val_hierarchy)

model2 = MLP(x_train.shape[1], len(np.unique(y_train)), [256, 128], dropout=dropout)

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





