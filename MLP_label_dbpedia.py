from typing import List, Iterable

import numpy as np
import torch as th
import pandas as pd
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from mlp_helper import csr_to_torch, select_relabel_documents, load_amazon, torch_to_csr, load_dbpedia
from textgcn.lib.models import MLP




CPU_ONLY = False
EARLY_STOPPING = False
epochs = 5
train_val_split = 0.1
lr = 0.05
dropout = 0.7
seed = 44
result_file = "results.csv"
model = MLP
np.random.seed(seed)
th.random.manual_seed(seed)
save_results = True

try:
    df = pd.read_csv(result_file)
except:
    df = pd.DataFrame(columns=["seed", "model", "hierarchy", "f1-macro", "accuracy"])
print("Loading data.")
(x_train, y_train), (x_test, y_test), (x_val, y_val) = load_dbpedia()
print("Training per-label approach for all categories.")

# First category (= flat)


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

        y_pred = np.argmax(logits_test.cpu().numpy(), axis=1) # get prediction for test set (for next category)

        f1_val = f1_score(y_val[0], pred_val, average='macro')
        acc_train = accuracy_score(y_train[0].cpu(), pred_train)
        print(f"[{epoch + 1:{length}}] loss: {loss.item(): .3f}, "
              f"training accuracy: {acc_train: .3f}, val_f1: {f1_val: .3f}")

# Train subsequent categories:


for cat in range(2): # 2 levels for dbpedia
    criterion = th.nn.CrossEntropyLoss(reduction='mean')
    device = th.device('cuda' if th.cuda.is_available() and not CPU_ONLY else 'cpu')
    encoders = []

    predictions = np.zeros_like(y_test[cat+1])

    for label in np.unique(y_train[cat].cpu()):
        print(f"Processing top level label {label}")

        # Filter labels

        (x_train_i, y_train_i), (x_val_i, y_val_i), le = \
            select_relabel_documents(x_train, x_val, y_train[cat+1], y_val[cat+1], y_train[cat], y_val[cat], label)

        encoders.append(le)
        x_train_i = x_train_i.to(device).float()
        y_train_i = th.from_numpy(y_train_i).to(device)
        x_val_i = x_val_i.to(device).float()

        model2 = MLP(x_train.shape[1], len(th.unique(y_train_i)), [256, 128], dropout=dropout)
        model2 = model2.to(device).float()

        print(f"x_val shape: {x_val.shape}")
        print(f"x_train shape: {x_train.shape}")
        print(f"y_train shape: {y_train[cat].shape}")
        # optimizer needs to be constructed AFTER the model was moved to GPU
        optimizer = th.optim.Adam(model2.parameters(), lr=lr)
        length = len(str(epochs))
        print(f"### Training start (Level {cat+1} Label {label})! ###")
        for epoch in range(epochs):
            model2.train()
            outputs = model2(x_train_i)
            loss = criterion(outputs, y_train_i)
            # performance tip: try set_to_none=True
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model2.eval()
            with th.no_grad():
                logits = model2(x_train_i)
                logits_val = model2(x_val_i)
                pred_val = np.argmax(logits_val.cpu().numpy(), axis=1)
                pred_train = np.argmax(logits.cpu().numpy(), axis=1)
                f1_val = f1_score(y_val_i, pred_val, average='macro')
                acc_train = accuracy_score(y_train_i.cpu(), pred_train)
                print(f"[{epoch + 1:{length}}] loss: {loss.item(): .3f}, "
                      f"training accuracy: {acc_train: .3f}, val_f1: {f1_val: .3f}")
                del logits, logits_val, pred_val, pred_train, f1_val, acc_train

        x_train_i, y_train_i, x_val_i = map(lambda x: x.cpu(), [x_train_i, y_train_i, x_val_i])

        # Prediction
        with th.no_grad():
            mask = y_pred == label
            masked_test = csr_to_torch(torch_to_csr(x_test)[mask])
            pred = np.argmax(model2(masked_test.to(device)).cpu(), axis=1)
            predictions[mask] = encoders[label].inverse_transform(pred)

        model2 = model2.cpu()

    print("Optimization finished!")


    y_pred = predictions # store predictions for next category

    acc_test = accuracy_score(y_test[cat+1], predictions)
    f1 = f1_score(y_test[cat+1], predictions, average='macro')

    print(f"Test Accuracy: {acc_test: .3f}")
    print(f"F1-Macro: {f1: .3f}")

#todo saving
if save_results:
    i = df.index.max() + 1 if df.index.max() != np.nan else 0
    df.loc[i] = {'seed': seed, 'model': "MLP", 'hierarchy': "per-level", 'f1-macro': f1,
                 'accuracy': acc_test}
    df.to_csv(result_file)



