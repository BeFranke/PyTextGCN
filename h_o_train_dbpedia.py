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
lable_category = "Cat2"
#Hyperparameters to optimize
#learning rate
lrs = [0.001, 0.005, 0.01, 0.05]
#dropout
dos = [0.5, 0.7]
# df_max
dfs = [0.5, 0.6, 0.7]

models = [GCN]

#n_hidden
n_hidden = 100

# resulting dataframe containing all result values
maxExpSize = len(lrs) * len(dos) * len(dfs) * len(models)
resultDf = pd.DataFrame(index= range(1, maxExpSize, 1),
                        columns= ["LR", "DO", "max_df", "model", "mean f1", "std f1"])
resultDf.fillna(0)


train = pd.read_csv("data/dbpedia/DBPEDIA_train.csv")
val = pd.read_csv("data/dbpedia/DBPEDIA_val.csv")
test = pd.read_csv("data/dbpedia/DBPEDIA_test.csv")

x = train['text'].tolist()
y = train[lable_category].tolist()

x_val = val['text'].tolist()
y_val = val['l3'].tolist()

val_idx = np.arange(len(x), len(x) + len(x_val))

x += x_val
y += y_val

x_test = test['text'].tolist()
y_test = test['text'].tolist()

test_idx = np.arange(len(x), len(x) + len(x_test))

x += x_test
y += y_test

y = LabelEncoder().fit_transform(y)
print("Data loaded!")


################################################  Text to Graph ################################################

kf = KFold(n_splits=k_split, shuffle=True)
frameIterator = 0
timestamp = datetime.now().strftime("%d_%b_%y_%H_%M_%S")
csv_name = "DBPEDIA_Flat_HypOpt_" + lable_category + "_" + timestamp + ".csv"

for mdf in dfs:
    t2g = Text2GraphTransformer(n_jobs=8, min_df=5, save_path=None, verbose=1, max_df=mdf)

    g = t2g.fit_transform(x, y, test_idx=test_idx, val_idx=val_idx)
    print("Graph built!")

    # Mask for doc nodes
    mask = th.logical_or(g.train_mask, g.test_mask)

    indizes = th.nonzero(mask)
    for dropout in dos:
        for lr in lrs:
            for model in models:
                model_name = "GCN" if model == GCN else "EGCN"
                try:
                    scores = np.zeros(k_split)
                    for i, (train, test) in enumerate(kf.split(indizes)):
                        train = train + g.n_vocab
                        test = test + g.n_vocab
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
                        print("[{}/{}] ----- split {} -- lr: {} -- do: {} - max_df: {}"
                              .format(frameIterator+1, maxExpSize, i, lr, dropout, mdf))
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
                                val_loss = criterion(logits[g.val_mask], g.y[g.val_mask])
                                pred_val = np.argmax(logits[g.val_mask].cpu().numpy(), axis=1)
                                pred_train = np.argmax(logits[g.val_mask].cpu().numpy(), axis=1)
                                f1_val = f1_score(g.y.cpu()[g.val_mask], pred_val, average="macro")
                                acc_train = accuracy_score(g.y.cpu()[g.val_mask], pred_train)
                            print(f"[{epoch + 1:{length}}] loss: {loss.item(): .3f}, "
                            f"training accuracy: {acc_train: .3f}, val_f1: {f1_val: .3f}")
                        scores[i] = f1_val

                    frameIterator += 1
                    # Dataframe structure: "LR", "DO", "n_hidden", "accuracy mean", "accuracy std"

                    resultDf.loc[frameIterator] = [lr, dropout, mdf, model_name, scores.mean(), scores.std()]
                except RuntimeError as e:
                    print("CUDA ran out of memory. Setting NaN.")
                    resultDf.loc[frameIterator] = [lr, dropout, mdf, model_name, np.nan, np.nan]
                resultDf.to_csv(csv_name, encoding='utf-8')


resultDf.to_csv(csv_name, encoding='utf-8')

print("Optimization finished!")

