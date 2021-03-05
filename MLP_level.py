import os
import os

import numpy as np
import pandas as pd
import torch as th
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder

from mlp_helper import load_amazon, load_dbpedia, append_feats
from textgcn.lib.models import MLP

CPU_ONLY = False
EARLY_STOPPING = False
epochs = 50
lr = 0.05
dropout = 0.7
seed = 44
result_file = "results_mlp.csv"
model = MLP
np.random.seed(seed)
th.random.manual_seed(seed)
save_results = True

# Dataset dependend settings
dataset = "dbpedia"
train_val_split = 0.1  # only for amazon

df = pd.DataFrame(columns=["seed", "dataset", "hierarchy", "category", "f1-macro", "accuracy"])

print("Loading data.")
if dataset == "amazon":
    categories = 2
    (x_train, y_train), (x_test, y_test), (x_val, y_val) = load_amazon(train_val_split=train_val_split)
else:
    dataset = "dbpedia"
    categories = 3
    (x_train, y_train), (x_test, y_test), (x_val, y_val) = load_dbpedia()

print("Training per-level approach for all categories.")




model1 = MLP(x_train.shape[1], len(np.unique(y_train[0])), [256, 128], dropout=dropout)
criterion = th.nn.CrossEntropyLoss(reduction='mean')

device = th.device('cuda' if th.cuda.is_available() and not CPU_ONLY else 'cpu')
model1 = model1.to(device).float()

x_train = x_train.to(device)
y_train[0] = y_train[0].to(device)
x_val = x_val.to(device)

x_test = x_test.to(device)
print(f"x_val shape: {x_val.shape}")
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train[0].shape}")
# optimizer needs to be constructed AFTER the model was moved to GPU
optimizer = th.optim.Adam(model1.parameters(), lr=lr)
length = len(str(epochs))
print("### Training start (Top-Level)! ###")
y_pred = None

for epoch in range(epochs):
    model1.train()
    outputs = model1(x_train)
    loss = criterion(outputs, y_train[0])
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
        logits_test = model1(x_test)

        y_pred = np.argmax(logits_test.cpu().numpy(), axis=1)  # get prediction for test set (for next category)

        f1_val = f1_score(y_val[0], pred_val, average='macro')
        acc_train = accuracy_score(y_train[0].cpu(), pred_train)
        print(f"[{epoch + 1:{length}}] loss: {loss.item(): .3f}, "
              f"training accuracy: {acc_train: .3f}, val_f1: {f1_val: .3f}")

for cat in range(categories - 1):
    criterion = th.nn.CrossEntropyLoss(reduction='mean')
    device = th.device('cuda' if th.cuda.is_available() and not CPU_ONLY else 'cpu')

    oh = OneHotEncoder(sparse=False)

    y_train[cat] = oh.fit_transform(y_train[cat].cpu().numpy().reshape(-1,1))
    y_val[cat] = oh.transform(y_val[cat].cpu().numpy().reshape(-1,1))
    y_test[cat] = oh.transform(y_test[cat].cpu().numpy().reshape(-1,1))
    y_pred = oh.transform(y_pred.reshape(-1,1)) # use previous predictions for test set


    x_train = append_feats(x_train, y_train[cat])
    x_val = append_feats(x_val, y_val[cat])
    x_test = append_feats(x_test, y_pred) # augment test data with predictions of previous layer

    model2 = MLP(x_train.shape[1], len(np.unique(y_train[cat+1])), [256, 128], dropout=dropout)

    criterion = th.nn.CrossEntropyLoss(reduction='mean')

    device = th.device('cuda' if th.cuda.is_available() and not CPU_ONLY else 'cpu')
    model2 = model2.to(device).float()
    x_train = x_train.to(device).float()
    y_train[cat+1] = y_train[cat+1].to(device)
    x_val = x_val.to(device).float()

    # optimizer needs to be constructed AFTER the model was moved to GPU
    optimizer = th.optim.Adam(model2.parameters(), lr=lr)
    length = len(str(epochs))
    print(f"### Training start (Level {cat})! ###")
    for epoch in range(epochs):
        model2.train()
        outputs = model2(x_train)
        loss = criterion(outputs, y_train[cat+1])
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
            f1_val = f1_score(y_val[cat+1].cpu(), pred_val, average='macro')
            acc_train = accuracy_score(y_train[cat+1].cpu(), pred_train)
            print(f"[{epoch + 1:{length}}] loss: {loss.item(): .3f}, "
                  f"training accuracy: {acc_train: .3f}, val_f1: {f1_val: .3f}")

    print("Optimization finished!")

    with th.no_grad():
        x_test = x_test.to(device).float()

        pred_test = np.argmax(model2(x_test).cpu(), axis=1)
        acc_test = accuracy_score(y_test[cat+1].cpu(), pred_test)

        f1 = f1_score(y_test[cat+1].cpu().detach(), pred_test, average='macro')
        y_pred = pred_test # store predictions to augment test data for next category

    print(f"Test Accuracy: {acc_test: .3f}")
    print(f"F1-Macro: {f1: .3f}")

    x_train = x_train.cpu()
    x_val = x_val.cpu()
    x_test = x_test.cpu()

    if save_results:
        df = df.append({'seed': seed, "dataset": dataset, 'hierarchy': "per-level", "category": cat,
                        'f1-macro': f1,
                        'accuracy': acc_test}, ignore_index=True)

if save_results:
    if not os.path.isfile(result_file):
        df.to_csv(result_file, index=False)
    else:
        df.to_csv(result_file, index=False, mode='a', header=False)




