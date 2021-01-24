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
epochs = 100
train_val_split = 0.1
k_split = 5

# lr = 0.2
save_model = False
label_category = "Cat1"
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
n_hidden_start = 16
n_hidden_stop = 16
n_hidden_step = 1
#
lr_size = ((lr_stop - lr_start) / lr_step + 1)
do_size = ((do_stop - do_start) / do_step + 1)
n_hidden_size = ((n_hidden_stop - n_hidden_start) / n_hidden_step + 1)
frameIterator = 0;
maxExpSize = lr_size*do_size*n_hidden_size

# resulting dataframe containing all result values
# 2lc = second label classifier
# LR = learning rate
# DO = Drop out rate
# mean acc = Mean accuracy on cross validation scores
# std acc = standart derivation on cross validation scores
resultDf = pd.DataFrame(index= range(1, int(maxExpSize + 1), 1),
                        columns= ["2lc", "LR", "DO", "n_hidden", "mean acc", "std acc"])
resultDf.fillna(0)


train = pd.read_csv("data/amazon/train.csv")

#save_path = "textgcn/graphs/"
save_path = None
x = train['Text'].tolist()
y_top = train['Cat1'].tolist()
y = train['Cat2'].tolist()

y_top = LabelEncoder().fit_transform(y_top)
y = LabelEncoder().fit_transform(y)

num_labels = len(np.unique(y_top))

# separated labels variable into 6 bitmap vectors
labels = np.zeros((num_labels, len(y_top)))
for i in np.arange(len(y_top)):
    labels[y_top[i]][i] = 1

# Train/val split
#test_idx = np.random.choice(len(x), int(train_val_split * len(x)), replace=False)

print("Data loaded!")

################################################  Text to Graph ################################################

t2g = Text2GraphTransformer(n_jobs=8, min_df=5, save_path=save_path, verbose=1, max_df=0.9)
#g = t2g.fit_transform(x, y, test_idx=test_idx)

for classifier in range(num_labels):

    # get index of all labels with category n
    available_labels = np.nonzero(labels[classifier])[0]
    # choose according to train/test split from all indices a certain amount at random
    #test_idx = np.random.choice(available_labels, int(train_val_split * len(available_labels)), replace=False)
    mask = y_top == classifier
    # build the graph on the basis of the chosen indices

    g = t2g.fit_transform(x, y, test_idx=[])
    # get the graph according to the classifier label
    #index = labels[classifier]
    #g.test_mask[:] = 0
    #g.test_mask[test_idx] = 1
    #g.train_mask[:] = 0
    # Mask for doc nodes
    #mask = th.logical_or(g.train_mask,  g.test_mask)


    kf = KFold(n_splits=k_split, shuffle=True)

    print(np.unique(g.y[g.test_mask], return_counts=True))
    print(np.unique(g.y, return_counts=True))
    indices = th.nonzero(th.from_numpy(mask)) + g.n_vocab


    # relabel the selected labels in ascending order
    #TODO suspected error here. Relabeling?
    g.y[g.train_mask] = th.from_numpy(LabelEncoder().fit_transform(g.y[g.train_mask]))


    for n_hidden in np.arange(n_hidden_start, n_hidden_stop+n_hidden_step, n_hidden_step):
        for dropout in np.arange(do_start, do_stop+do_step, do_step):
            for lr in np.arange(lr_start, lr_stop + lr_step, lr_step):
                scores = np.zeros(k_split)
                for i, (train, test) in enumerate(kf.split(indices)):
                    g.test_mask[:] = 0
                    g.test_mask[indices[test]] = 1
                    g.train_mask[:] = 0
                    g.train_mask[indices[train]] = 1
                ################################################  define GCN  ################################################
                    # gcn = JumpingGCN(g.x.shape[1], len(np.unique(y)), n_hidden_gcn=32)
                    #gcn = GCN(g.x.shape[1], len(np.unique(y)), n_hidden_gcn=n_hidden, dropout=dropout)
                    gcn = GCN(g.x.shape[1], len(np.unique(g.y[g.train_mask])), n_hidden_gcn=n_hidden, dropout=dropout)
                    # gcn = HierarchyGNN(in_feats=g.x.shape[1], n_classes=len(np.unique(y)), n_hidden=64, mlp_hidden=0, mlp_layers=1, graph_layer=nn.GraphConv)
                    # gcn = JumpingKnowledgeNetwork(g.x.shape[1], len(np.unique(y)), n_hidden_gcn=n_hidden, dropout=dropout, activation=th.nn.SELU)

                    criterion = th.nn.CrossEntropyLoss(reduction='mean')

                    device = th.device('cuda' if th.cuda.is_available() and not CPU_ONLY else 'cpu')
                    gcn = gcn.to(device).float()
                    g = g.to(device)


                    # optimizer needs to be constructed AFTER the model was moved to GPU
                    optimizer = th.optim.Adam(gcn.parameters(), lr=lr)

                ################################################  Training  ################################################
                    length = len(str(epochs))
                    print("[{}/{}] ----- split {} -- lr: {} -- do: {} - n_hidden: {}".format(frameIterator+1,maxExpSize,i ,lr,dropout,n_hidden))
                    time_start = datetime.now()

                    for epoch in range(epochs):
                        gcn.train()
                        outputs = gcn(g)[g.train_mask]
                        # TODO g.y contains out of range labels
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
                classifier_name = "classifier_" + classifier
                resultDf.loc[frameIterator] = [classifier_name, lr, dropout, n_hidden, scores.mean(), scores.std()]

timestamp = datetime.now().strftime("%d_%b_%y_%H_%M_%S")
csv_name = "HypOpt_Labels_" + label_category + "_" + timestamp + ".csv"
resultDf.to_csv(csv_name, encoding='utf-8')

print("Optimization finished!")

