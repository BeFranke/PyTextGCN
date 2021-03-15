import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from textgcn import Text2GraphTransformer
from textgcn.lib.models import *

"""
this is just flat_amazon.py without the training part to benchmark the graph building
"""

CPU_ONLY = False
EARLY_STOPPING = False
epochs = 500
train_val_split = 0.1
lr = 0.05
save_model = False
dropout = 0.7
max_df = 0.7
seed = 44
result_file = "results.csv"
model = GCN
np.random.seed(seed)
th.random.manual_seed(seed)
save_results = True
labels = "Cat2"
window_size=20

try:
    df = pd.read_csv(result_file)
except:
    df = pd.DataFrame(columns=["seed", "model", "hierarchy", "f1-macro", "accuracy"])

train = pd.read_csv("data/amazon/train.csv")
test = pd.read_csv("data/amazon/test.csv")

save_path = "textgcn/graphs/"
# save_path = None
x = train['Text'].tolist()
y = train[labels].tolist()

# Train/val split
val_idx = np.random.choice(len(x), int(train_val_split * len(x)), replace=False)
train_idx = np.array([x for x in range(len(x)) if x not in val_idx])

x_test = test['Text'].tolist()
y_test = test[labels].tolist()

test_idx = np.arange(len(x), len(x) + len(x_test))

# Combine training & test data set
x = x + x_test
y = y + y_test

y = LabelEncoder().fit_transform(y)
print("Data loaded!")

t2g = Text2GraphTransformer(n_jobs=8, min_df=5, save_path=None, verbose=1, max_df=max_df, window_size=window_size)
# t2g = Text2GraphTransformer(n_jobs=8, min_df=1, save_path=save_path, verbose=1, max_df=1.0)
ls = os.listdir("textgcn/graphs")
# if not ls:
if True:
    g = t2g.fit_transform(x, y, test_idx=test_idx, val_idx=val_idx)
    print("Graph built!")
else:
    g = t2g.load_graph(os.path.join(save_path, ls[0]))
    print(f"Graph loaded from {os.path.join(save_path, ls[0])}!")
    print(f"n_classes={len(np.unique(g.y))}")

