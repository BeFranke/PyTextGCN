import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch as th
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from textgcn import GCN, Text2GraphTransformer

CPU_ONLY = False
EARLY_STOPPING = False
epochs = 500
train_val_split = 0.1

train = pd.read_csv("data/amazon/train.csv")
test = pd.read_csv("data/amazon/test.csv")

save_path = "textgcn/graphs/"
# save_path = None
x = train['Text'].tolist()
y = train['Cat1'].tolist()

# Train/val split
val_idx = np.random.choice(len(x), int(train_val_split * len(x)), replace=False)
train_idx = np.array([x for x in range(len(x)) if x not in val_idx])

x_test = test['Text'].tolist()
y_test = test['Cat1'].tolist()

test_idx = np.arange(len(x)+1, len(x)+ len(x_test))

# Combine training & test data set
x = x + x_test
y = y + y_test

y = LabelEncoder().fit_transform(y)
print("Data loaded!")

t2g = Text2GraphTransformer(n_jobs=8, min_df=5, save_path=save_path, verbose=1, max_df=0.9)
ls = os.listdir("textgcn/graphs")
if not ls:
    g = t2g.fit_transform(x, y, test_idx=test_idx, val_idx=val_idx)
else:
    g = t2g.load_graph(os.path.join("textgcn/graphs", ls[0]))

print("Graph built")

gcn = GCN(g.x.shape[1], len(np.unique(y)), n_hidden_gcn=50)

criterion = th.nn.CrossEntropyLoss(reduction='mean')

device = th.device('cuda' if th.cuda.is_available() and not CPU_ONLY else 'cpu')
gcn = gcn.to(device).float()
g = g.to(device)

# optimizer needs to be constructed AFTER the model was moved to GPU
optimizer = th.optim.Adam(gcn.parameters(), lr=0.02)
history = []
length = len(str(epochs))
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
        pred_val = np.argmax(gcn(g)[g.val_mask].cpu().numpy(), axis=1)
        pred_train = np.argmax(gcn(g)[g.train_mask].cpu().numpy(), axis=1)
        acc_val = accuracy_score(g.y.cpu()[g.val_mask].detach(), pred_val)
        acc_train = accuracy_score(g.y.cpu()[g.train_mask].detach(), pred_train)
        print(f"[{epoch + 1:{length}}] loss: {loss.item(): .3f}, "
              f"training accuracy: {acc_train: .3f}, val_accuracy: {acc_val: .3f}")
    history.append((loss.item(), acc_val))
    if epoch > 5 and history[-5][0] < history[-1][0] and EARLY_STOPPING:
        print("early stopping activated!")
        break

print("Optimization finished!")
print("Saving model...")
th.save(gcn, f"models/gcn_{int(time.time())}.nn")
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
print(f"Training took {time_end - time_start} for {epoch} epochs.")

loss, acc = zip(*history)

fig, axs = plt.subplots(2)
axs[0].plot(loss, label="TrainLoss")
axs[1].plot(acc, label="ValAcc")
axs[0].legend()
axs[1].legend()
plt.show()
