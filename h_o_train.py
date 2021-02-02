import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch as th
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch_geometric import nn

from textgcn import Text2GraphTransformer
from textgcn.lib.models import *

CPU_ONLY = False
epochs = 500
train_val_split = 0.1
k_split = 3

# lr = 0.2
save_model = False
lable_category = "Cat1"
#Hyperparameters to optimize
#learning rate
lrs = [0.0001, 0.001, 0.01, 0.1]
#dropout
dos = [0.5, 0.7]
# df_max
dfs = [0.5, 0.7, 0.9]

models = [GCN, EGCN]

#n_hidden
n_hidden = 100

# resulting dataframe containing all result values
maxExpSize = len(lrs) * len(dos) * len(dfs) * len(models)
resultDf = pd.DataFrame(index= range(1, maxExpSize, 1),
                        columns= ["LR", "DO", "max_df", "mean acc", "std acc"])
resultDf.fillna(0)


train = pd.read_csv("data/amazon/train.csv")

#save_path = "textgcn/graphs/"
save_path = None
x = train['Text'].tolist()
y = train[lable_category].tolist()

# Train/val split
test_idx = np.random.choice(len(x), int(train_val_split * len(x)), replace=False)

y = LabelEncoder().fit_transform(y)
print("Data loaded!")


################################################  Text to Graph ################################################

t2g = Text2GraphTransformer(n_jobs=8, min_df=5, save_path=save_path, verbose=1, max_df=0.9)

g = t2g.fit_transform(x, y, test_idx=test_idx)
print("Graph built!")

# Mask for doc nodes
mask = th.logical_or(g.train_mask,  g.test_mask)

indizes  = th.nonzero(mask)
kf = KFold(n_splits=k_split, shuffle=True)
frameIterator = 0

for mdf in dfs:
    for dropout in dos:
        for lr in lrs:
            for model in models:
                try:
                    scores = np.zeros(k_split)
                    for i, (train, test) in enumerate(kf.split(indizes)):
                        g.test_mask[:] = 0
                        g.test_mask[test] = 1
                        g.train_mask[:] = 0
                        g.train_mask[train] = 1
                        ################################################  define GCN  ################################
                        gcn = model(g.x.shape[1], len(np.unique(y)), n_hidden_gcn=n_hidden, dropout=dropout)

                        criterion = th.nn.CrossEntropyLoss(reduction='mean')
                        device = th.device('cuda' if th.cuda.is_available() and not CPU_ONLY else 'cpu')
                        gcn = gcn.to(device).float()
                        g = g.to(device)


                        # optimizer needs to be constructed AFTER the model was moved to GPU
                        optimizer = th.optim.Adam(gcn.parameters(), lr=lr)

                        ###################################  Training  ######################################
                        length = len(str(epochs))
                        print("[{}/{}] ----- split {} -- lr: {} -- do: {} - n_hidden: {}"
                              .format(frameIterator+1, maxExpSize, i, lr, dropout, n_hidden))
                        time_start = datetime.now()

                        for epoch in range(epochs):
                            gcn.train()
                            outputs = gcn(g)[g.train_mask]
                            loss = criterion(outputs, g.y[g.train_mask])
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
                except RuntimeError as e:
                    print("CUDA ran out of memory. Setting NaN.")
                    resultDf.loc[frameIterator] = [lr, dropout, n_hidden, np.nan, np.nan]

timestamp = datetime.now().strftime("%d_%b_%y_%H_%M_%S")
csv_name = "Lvl_HypOpt_" + lable_category + "_" + timestamp + ".csv"
resultDf.to_csv(csv_name, encoding='utf-8')

print("Optimization finished!")

