import os
import time
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import torch as th
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch_geometric import nn

from textgcn import Text2GraphTransformer
from textgcn.lib.models import *


mode_hparams = {
    'flat': {'lr': 0.025, 'dropout': 0.5},
    'lvl1': {'lr': 0.07, 'dropout': 0.5},
    'lvl2': {'lr': 0.2, 'dropout': 0.5}
    # to be extended
}

N_HIDDEN = 64
cpu = th.device('cpu')
gpu = th.device('cuda')

def prepare_data(mode, test_hierarchy=None):
    assert mode in ["flat", "lvl1", "lvl2"] + [f"lbl{i}" for i in range(6)]
    train = pd.read_csv("data/amazon/train.csv")
    test = pd.read_csv("data/amazon/test.csv")
    x = train['Text'].tolist()
    train_idx = np.arange(0, len(x))
    x += test['Text'].tolist()
    test_idx = np.arange(len(train_idx), len(x))
    if mode == "flat" or "lbl" in mode:
        y = train['Cat2'].tolist() + test['Cat2'].tolist()
        y_top = None
        y = LabelEncoder().fit_transform(y)
        hierarchy = None
    elif mode == "lvl1":
        y = train['Cat1'].tolist() + test['Cat1'].tolist()
        y_top = None
        y = LabelEncoder().fit_transform(y)
        hierarchy = None
    elif mode == "lvl2":
        y = train['Cat2'].tolist() + test['Cat2'].tolist()
        y_top = train['Cat1'].tolist() + test['Cat1'].tolist()
        y = LabelEncoder().fit_transform(y)
        y_top = LabelEncoder().fit_transform(y_top)
        if test_hierarchy is None:
            hierarchy = OneHotEncoder(sparse=False).fit_transform(y_top.reshape(-1, 1))
        else:
            hierarchy = test_hierarchy
    else:
        raise Exception

    t2g = Text2GraphTransformer(n_jobs=8, min_df=5, verbose=1, max_df=0.9)
    g = t2g.fit_transform(x, y, test_idx=test_idx, val_idx=None, hierarchy_feats=hierarchy)

    return g


def train_test(g, lr, dropout, epochs, save_model=True, fname=None, device=th.device('cuda'),
               n_hidden=N_HIDDEN):
    mask = th.logical_or(g.train_mask, g.test_mask)
    gcn = GCN(g.x.shape[1], len(np.unique(g.y[mask])), n_hidden_gcn=n_hidden, dropout=dropout)

    criterion = th.nn.CrossEntropyLoss(reduction='mean')

    gcn = gcn.to(device).float()
    g = g.to(device)

    # optimizer needs to be constructed AFTER the model was moved to GPU
    optimizer = th.optim.Adam(gcn.parameters(), lr=lr)
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
            pred_train = np.argmax(logits[g.train_mask].cpu().numpy(), axis=1)
            acc_train = accuracy_score(g.y.cpu()[g.train_mask], pred_train)
            print(f"[{epoch + 1:{length}}] loss: {loss.item(): .3f}, "
                  f"training accuracy: {acc_train: .3f}")

    print("Optimization finished!")
    if save_model:
        print("Saving model...")
        if fname is None:
            th.save(gcn, f"models/gcn_{int(time.time())}.nn")
        else:
            th.save(gcn, f"models/gcn_{fname}.nn")

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
    return f1, gcn


def eval_flat(results: Dict):
    g = prepare_data("flat")
    f1, gcn = train_test(g, epochs=200, fname="flat", **mode_hparams['flat'])
    # make sure to clean the GPU after usage
    g = g.to(cpu)
    gcn = gcn.to(cpu)
    results['mode'].append("flat")
    results['f1'].append(f1)


def eval_lvl(results: Dict):
    g1 = prepare_data("lvl1")
    _, gcn1 = train_test(g1, epochs=200, fname="lvl1", **mode_hparams['lvl1'])
    g1 = g1.to(cpu)
    gcn1 = gcn1.to(cpu)
    g2 = prepare_data("lvl2")
    _, gcn2 = train_test(g2, epochs=200, fname="lvl2", **mode_hparams['lvl2'])
    g2 = g2.to(cpu)
    del g2
    gcn2 = gcn2.to(cpu)

    # now evaluate with predicted lvl1 labels as input
    gcn1 = gcn1.to(gpu)
    gcn1.eval()
    g1 = g1.to(gpu)
    pred = gcn1(g1)     # lets just use the raw logits
    pred = pred[g1.n_vocab:]
    g2 = prepare_data("lvl2", pred.to(cpu))
    g2 = g2.to(gpu)
    gcn2 = gcn2.cuda()
    gcn2.eval()
    with th.no_grad():
        pred_test = np.argmax(gcn2(g2)[g2.test_mask].cpu().detach().numpy(), axis=1)
        acc_test = accuracy_score(g2.y.cpu()[g2.test_mask].detach(), pred_test)
        f1 = f1_score(g2.y.cpu()[g2.test_mask].detach(), pred_test, average='macro')
        conf_mat = confusion_matrix(g2.y.cpu()[g2.test_mask].detach(), pred_test)
    print(f"Test Accuracy: {acc_test: .3f}")
    print(f"F1-Macro: {f1: .3f}")
    print("Confusion matrix:")
    print(conf_mat)
    results['mode'].append("lvl")
    results['f1'].append(f1)


if __name__ == '__main__':
    res = {'mode': [], 'f1': []}
    eval_flat(res)
    eval_lvl(res)
    print(pd.DataFrame(res))




