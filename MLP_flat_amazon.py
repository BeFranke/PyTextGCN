from typing import List

import numpy as np
import torch as th
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

def csr_to_torch(csr):
    acoo = csr.tocoo()
    return th.sparse_coo_tensor(th.LongTensor([acoo.row.tolist(), acoo.col.tolist()]),
                                th.FloatTensor(acoo.data.astype(np.float32)),
                                size=csr.shape)


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
epochs = 500
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

# Train/val split
val_idx = np.random.choice(len(raw), int(train_val_split * len(raw)), replace=False)
val_mask = np.zeros(len(raw), dtype=np.bool)
val_mask[val_idx] = 1
raw_val = raw[val_mask]
raw_train = raw[np.logical_not(val_mask)]

y_val = y_train[val_mask]
y_train = y_train[np.logical_not(val_mask)]

tfidf = TfidfVectorizer(stop_words='english', max_df=0.9)
x_train = tfidf.fit_transform(raw_train)
y_train = y_train.tolist()
x_val = tfidf.transform(raw_val.tolist())


raw_test = test['Text'].tolist()
y_test = test[labels].tolist()

x_test = tfidf.transform(raw_test)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.transform(y_val)
y_test = le.transform(y_test)

print("Data loaded!")

del raw
del raw_train
del raw_val
del raw_test

x_train, x_val, x_test = map(csr_to_torch, [x_train, x_val, x_test])
y_train, y_val, y_test = map(th.from_numpy, [y_train, y_val, y_test])

model = MLP(x_train.shape[1], len(np.unique(y_train)), [256, 128], dropout=dropout)

criterion = th.nn.CrossEntropyLoss(reduction='mean')

device = th.device('cuda' if th.cuda.is_available() and not CPU_ONLY else 'cpu')
model = model.to(device).float()
x_train = x_train.to(device)
y_train = y_train.to(device)
x_val = x_val.to(device)

print(f"x_val shape: {x_val.shape}")
print(f"x_train shape: {x_train.shape}")

# optimizer needs to be constructed AFTER the model was moved to GPU
optimizer = th.optim.Adam(model.parameters(), lr=lr)
length = len(str(epochs))
print("### Training start! ###")
for epoch in range(epochs):
    model.train()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    # performance tip: try set_to_none=True
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    model.eval()
    with th.no_grad():
        logits = model(x_train)
        logits_val = model(x_val)
        pred_val = np.argmax(logits_val.cpu().numpy(), axis=1)
        pred_train = np.argmax(logits.cpu().numpy(), axis=1)
        f1_val = f1_score(y_val.cpu(), pred_val, average='macro')
        acc_train = accuracy_score(y_train.cpu(), pred_train)
        print(f"[{epoch + 1:{length}}] loss: {loss.item(): .3f}, "
              f"training accuracy: {acc_train: .3f}, val_f1: {f1_val: .3f}")

print("Optimization finished!")

with th.no_grad():
    x_test = x_test.to(device)
    pred_test = np.argmax(model(x_test).cpu().detach().numpy(), axis=1)
    acc_test = accuracy_score(y_test.cpu().detach(), pred_test)
    f1 = f1_score(y_test.cpu().detach(), pred_test, average='macro')

print(f"Test Accuracy: {acc_test: .3f}")
print(f"F1-Macro: {f1: .3f}")


if save_results:
    i = df.index.max() + 1 if df.index.max() != np.nan else 0
    df.loc[i] = {'seed': seed, 'model': "MLP", 'hierarchy': "flat", 'f1-macro': f1,
                 'accuracy': acc_test}
    df.to_csv(result_file)





