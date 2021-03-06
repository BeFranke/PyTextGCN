from typing import List

import numpy as np
import torch as th
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

from mlp_helper import load_dbpedia, load_amazon
from textgcn.lib.models import MLP
import os

CPU_ONLY = False
EARLY_STOPPING = True
patience = 10
min_epochs = 30

epochs = 500
lr = 2e-3
dropout = 0.5
seed = 44
result_file = "results_mlp.csv"
model = MLP
np.random.seed(seed)
th.random.manual_seed(seed)
save_results = True

# dataset dependend settings
dataset = "amazon"
train_val_split = 0.1 #only for amazon

df = pd.DataFrame(columns=["seed", "dataset", "hierarchy", "category", "f1-macro", "accuracy"])

print("Loading data.")
if dataset == "amazon":
    categories = 2
    (x_train, y_train), (x_test, y_test), (x_val, y_val) = load_amazon(train_val_split=train_val_split)
else:
    dataset = "dbpedia"
    categories = 3
    (x_train, y_train), (x_test, y_test), (x_val, y_val) = load_dbpedia()

print("Training per-label approach for all categories.")

for category in range(categories):
    history = []

    print(f"Training on category: {category}")
    y_train_c = y_train[category]
    y_test_c = y_test[category]
    y_val_c = y_val[category]

    model = MLP(x_train.shape[1], len(np.unique(y_train_c)), [256, 128], dropout=dropout)

    criterion = th.nn.CrossEntropyLoss(reduction='mean')

    device = th.device('cuda' if th.cuda.is_available() and not CPU_ONLY else 'cpu')
    model = model.to(device).float()
    x_train = x_train.to(device)
    y_train_c = y_train_c.to(device)
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
        loss = criterion(outputs, y_train_c)
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
            f1_val = f1_score(y_val_c.cpu(), pred_val, average='macro')
            acc_train = accuracy_score(y_train_c.cpu(), pred_train)
            print(f"[{epoch + 1:{length}}] loss: {loss.item(): .3f}, "
                  f"training accuracy: {acc_train: .3f}, val_f1: {f1_val: .3f}")

        history.append((loss.item(), f1_val))

        # scheduler.step(val_loss)

        if epoch > min_epochs and EARLY_STOPPING:
            dec_steps = 0
            for i in range(patience):
                dec_steps += (history[-(i+1)][1] <= history[-(patience+1)][1])
            if dec_steps >= patience:
                print(f"Early stopping! Validation f1 decreased for {dec_steps} epochs!")
                break


    print("Optimization finished!")

    with th.no_grad():
        x_test = x_test.to(device)
        pred_test = np.argmax(model(x_test).cpu().detach().numpy(), axis=1)
        acc_test = accuracy_score(y_test_c.cpu().detach(), pred_test)
        f1 = f1_score(y_test_c.cpu().detach(), pred_test, average='macro')

    x_train = x_train.cpu() # back to cpu
    x_val = x_val.cpu()
    x_test = x_test.cpu()

    print(f"Test Accuracy: {acc_test: .3f}")
    print(f"F1-Macro: {f1: .3f}")

    if save_results:
        df = df.append({'seed': seed, "dataset": dataset, 'hierarchy': 'flat', 'category': category,
                     'f1-macro': f1,
                     'accuracy': acc_test}, ignore_index=True)

if save_results:
    if not os.path.isfile(result_file):
        df.to_csv(result_file, index=False)
    else:
        df.to_csv(result_file, index=False, mode='a', header=False)