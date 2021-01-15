import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch as th
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch_geometric import nn

from textgcn import Text2GraphTransformer
from textgcn.lib.models import JumpingKnowledgeNetwork, GCN

CPU_ONLY = False
epochs = 100
train_val_split = 0.1
k_split = 5


lable_category = "Cat2"
#Hyperparameters to optimize
#learning rate
lr_start = 0.005
lr_stop = 0.5
lr_step = 0.005
#dropout
do_start = 0.05
do_stop = 0.7
do_step = 0.05
#n_hidden
n_hidden_start = 64
n_hidden_stop = 64
n_hidden_step = 1
#
lr_size = ((lr_stop - lr_start) / lr_step + 1)
do_size = ((do_stop - do_start) / do_step + 1)
n_hidden_size = ((n_hidden_stop - n_hidden_start) / n_hidden_step + 1)
frameIterator = 0
maxExpSize = lr_size*do_size*n_hidden_size

# set to "false" to evaluate flat approach
hierarchical_feats = True

# resulting dataframe containing all result values
resultDf = pd.DataFrame(index= range(1, int(maxExpSize + 1), 1), columns= ["LR", "DO", "n_hidden", "mean acc", "std acc"])
resultDf.fillna(0)

train = pd.read_csv("data/amazon/train.csv")
# test = pd.read_csv("data/amazon/test.csv")

# save_path = "textgcn/graphs/"
save_path = None
x = train['Text'].tolist()
y = train['Cat2'].tolist()
y_top = train['Cat1'].tolist()

# Train/val split
train_idx = np.random.choice(len(x), int(train_val_split * len(x)), replace=False)

y = LabelEncoder().fit_transform(y)
y_top = LabelEncoder().fit_transform(y_top)
print("Data loaded!")


################################################  Text to Graph ################################################
t2g = Text2GraphTransformer(n_jobs=8, min_df=5, save_path=save_path, verbose=1, max_df=0.9)
hierarchy = OneHotEncoder(sparse=False).fit_transform(y_top.reshape(-1, 1))


g = t2g.fit_transform(x, y, test_idx=train_idx, val_idx=None, hierarchy_feats=hierarchy if hierarchical_feats else None)
print("Graph built!")


# Mask for doc nodes
mask = th.logical_or(g.train_mask,  g.test_mask)

indizes  = th.nonzero(mask)
kf = KFold(n_splits=k_split, shuffle=True)

################################################  Build GCN ################################################

for n_hidden in np.arange(n_hidden_start, n_hidden_stop+n_hidden_step, n_hidden_step):
    for dropout in np.arange(do_start, do_stop+do_step, do_step):
        for lr in np.arange(lr_start, lr_stop + lr_step, lr_step):
            scores = np.zeros(k_split)
            for i, (train, test) in enumerate(kf.split(indizes)):
                g.test_mask[:] = 0
                g.test_mask[test] = 1
                g.train_mask[:] = 0
                g.train_mask[train] = 1

                gcn = GCN(g.x.shape[1], len(np.unique(y)), n_hidden_gcn=n_hidden, n_gcn=2)

                criterion = th.nn.CrossEntropyLoss(reduction='mean')

                device = th.device('cuda' if th.cuda.is_available() and not CPU_ONLY else 'cpu')
                gcn = gcn.to(device).float()
                g = g.to(device)

                # optimizer needs to be constructed AFTER the model was moved to GPU
                optimizer = th.optim.Adam(gcn.parameters(), lr=lr)
                scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, cooldown=10, verbose=True)
                length = len(str(epochs))
                print("[{}/{}] ----- split {} -- lr: {} -- do: {} - n_hidden: {}".format(frameIterator+1, maxExpSize, i, lr, dropout, n_hidden))
                time_start = datetime.now()
                for epoch in range(epochs):
                    gcn.train()
                    # TODO Cuda fehler "RuntimeError: CUDA out of memory. Tried to allocate 2.42 GiB (GPU 0; 8.00 GiB total capacity; 3.03 GiB already allocated; 1.05 GiB free; 5.37 GiB reserved in total by PyTorch)" in the line below
                    outputs = gcn(g)[g.train_mask]
                    loss = criterion(outputs, g.y[g.train_mask])
                    # performance tip: try set_to_none=True
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    gcn.eval()
                    with th.no_grad():
                        logits = gcn(g)
                        val_loss = criterion(logits[g.test_mask], g.y[g.test_mask])
                        pred_val = np.argmax(logits[g.test_mask].cpu().numpy(), axis=1)
                        pred_train = np.argmax(logits[g.train_mask].cpu().numpy(), axis=1)
                        acc_val = accuracy_score(g.y.cpu()[g.test_mask], pred_val)
                        acc_train = accuracy_score(g.y.cpu()[g.train_mask], pred_train)
                        print(f"[{epoch + 1:{length}}] loss: {loss.item(): .3f}, "
                              f"training accuracy: {acc_train: .3f}, val_accuracy: {acc_val: .3f}")
                scores[i] = acc_val
            frameIterator += 1
            # Dataframe structure: "LR", "DO", "n_hidden", "accuracy mean", "accuracy std"
            resultDf.loc[frameIterator] = [lr, dropout, n_hidden, scores.mean(), scores.std()]

print("Optimization finished!")

timestamp = datetime.now().strftime("%d_%b_%y_%H_%M_%S")
csv_name = "Hierarchical_HypOpt_" + lable_category + "_" + timestamp + ".csv"
resultDf.to_csv(csv_name, encoding='utf-8')

