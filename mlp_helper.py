from typing import List

import numpy as np
import torch as th
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy import sparse as sp
from textgcn.lib.models import MLP


def csr_to_torch(csr):
    acoo = csr.tocoo()
    return th.sparse_coo_tensor(th.LongTensor([acoo.row.tolist(), acoo.col.tolist()]),
                                th.FloatTensor(acoo.data.astype(np.float32)),
                                size=csr.shape)


def load_amazon(cats=None, train_val_split=0.2):
    if cats is None:
        cats = ["Cat1", "Cat2"]
    train = pd.read_csv("data/amazon/train.csv")
    test = pd.read_csv("data/amazon/test.csv")

    save_path = "textgcn/graphs/"
    # save_path = None
    raw = train['Text']

    # Train/val split
    val_idx = np.random.choice(len(raw), int(train_val_split * len(raw)), replace=False)
    val_mask = np.zeros(len(raw), dtype=np.bool)
    val_mask[val_idx] = 1
    raw_val = raw[val_mask]
    raw_train = raw[np.logical_not(val_mask)]

    # Transform x
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.9)
    x_train = tfidf.fit_transform(raw_train)
    x_val = tfidf.transform(raw_val.tolist())
    raw_test = test['Text'].tolist()
    x_test = tfidf.transform(raw_test)

    # Transform y
    ys_train = []
    ys_val = []
    ys_test = []

    for cat in cats:
        y_train = train[cat]
        y_val = y_train[val_mask]
        y_train = y_train[np.logical_not(val_mask)]
        y_train = y_train.tolist()
        y_test = test[cat].tolist()

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_val = le.transform(y_val)
        y_test = le.transform(y_test)
        y_train, y_val, y_test = map(th.from_numpy, [y_train, y_val, y_test])
        ys_train.append(y_train)
        ys_val.append(y_val)
        ys_test.append(y_test)


    del raw
    del raw_train
    del raw_val
    del raw_test

    x_train, x_val, x_test = map(csr_to_torch, [x_train, x_val, x_test])


    return (x_train, ys_train), (x_test, ys_test), (x_val, ys_val)


def load_dbpedia(cats=None):
    if cats is None:
        cats = ["l1", "l2", "l3"]
    train = pd.read_csv("data/dbpedia/DBPEDIA_train.csv")
    val = pd.read_csv("data/dbpedia/DBPEDIA_val.csv")
    test = pd.read_csv("data/dbpedia/DBPEDIA_test.csv")

    # save_path = None
    raw_train = train['text']
    # Train/val split
    raw_val = val['text']
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.9)
    x_train = tfidf.fit_transform(raw_train)
    x_val = tfidf.transform(raw_val.tolist())
    raw_test = test['text'].tolist()
    x_test = tfidf.transform(raw_test)


    del raw_train
    del raw_val
    del raw_test

    ys_train = []
    ys_val = []
    ys_test = []
    for cat in cats:
        y_train = train[cat]
        y_val = val[cat]
        y_train = y_train.tolist()
        y_val = y_val.tolist()
        y_test = test[cat].tolist()

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_val = le.transform(y_val)
        y_test = le.transform(y_test)
        y_train, y_val, y_test = map(th.from_numpy, [y_train, y_val, y_test])
        ys_train.append(y_train)
        ys_val.append(y_val)
        ys_test.append(y_test)

    x_train, x_val, x_test = map(csr_to_torch, [x_train, x_val, x_test])
    return (x_train, ys_train), (x_test, ys_test), (x_val, ys_val)


def torch_to_csr(X):
    X = X.coalesce().cpu()
    row, col = X.indices().numpy()
    data = X.values().numpy()
    shape = tuple(X.size())
    return sp.csr_matrix((data, (row, col)), shape=shape)


def select_relabel_documents(x_train, x_val, y_train, y_val, y_top_train, y_top_val, y_top_i):
    mask_train = (y_top_train == y_top_i).cpu()
    mask_val = y_top_val == y_top_i
    le = LabelEncoder()
    y_train_out = le.fit_transform(y_train[mask_train])
    y_val_out = le.transform(y_val[mask_val])
    x_train_csr = torch_to_csr(x_train)
    x_val_csr = torch_to_csr(x_val)
    return (csr_to_torch(x_train_csr[mask_train]), y_train_out), (csr_to_torch(x_val_csr[mask_val]), y_val_out), le


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
