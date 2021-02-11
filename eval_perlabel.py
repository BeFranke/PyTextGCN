import json

import torch as th
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from textgcn import Text2GraphTransformer

train_val_split = 0.1

with open("models/amazon/lvl1") as f:
    gcn1 = th.load(f)

gcn2s = []
for i in range(6):
    with open(f"models/amazon/lvl2-cat{i}") as f:
        gcn2s += [th.load(f)]

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

t2g = Text2GraphTransformer(n_jobs=8, min_df=5, save_path=save_path, verbose=1, max_df=0.6)

g1 = t2g.fit_transform(x, y_top, test_idx=test_idx, val_idx=val_idx, hierarchy_feats=None)

with th.no_grad():
    pred_test = np.argmax(gcn1(g1)[g1.test_mask].cpu().detach().numpy(), axis=1)

g2 = t2g.fit_transform(x, y, test_idx=test_idx, val_idx=val_idx, hierarchy_feats=None)

predictions = np.zeros(len(y))
predictions[:] = -1

with open("models/amazon/class_mapping.json") as f:
    mapping = json.load(f)

for i, gcn in enumerate(gcn2s):
    mask = th.logical_and((g1.y == i), g2.test_mask)
    pred = np.argmax(gcn(g2)[mask].cpu().detach().numpy(), axis=1)
    for j in range(len(pred)):
        pred[j] = mapping[i][pred[j]]
    predictions[mask] = pred

acc = accuracy_score(g2.y[g2.test_mask], predictions)
f1 = f1_score(g2.y[g2.test_mask], predictions, average="macro")

print(f"test accuracy: {acc}")
print(f"test f1-macro: {f1}")
