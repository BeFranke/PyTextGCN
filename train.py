import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch as th
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from textgcn import GCN, Text2GraphTransformer

CPU_ONLY = True
EARLY_STOPPING = False
epochs = 200

train = pd.read_csv("data/amazon/train_40k.csv")
# save_path="textgcn/graphs/"
save_path = None

x = train['Text'].tolist()
y = train['Cat1'].tolist()

y = LabelEncoder().fit_transform(y)

test_idx = np.random.choice(len(x), int(0.1 * len(x)), replace=False)
train_idx = np.array([x for x in range(len(x)) if x not in test_idx])
print("Data loaded!")

t2g = Text2GraphTransformer(n_jobs=8, min_df=5, save_path=save_path, verbose=1, max_df=0.9)
ls = os.listdir("textgcn/graphs")
if not ls:
    g = t2g.fit_transform(x, y, test_idx=test_idx)
else:
    g = t2g.load_graph(os.path.join("textgcn/graphs", ls[0]))

print("Graph built")

gcn = GCN(g.x.shape[1], len(np.unique(y)), n_hidden_gcn=200)

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
        predictions = np.argmax(gcn(g)[g.test_mask].cpu().numpy(), axis=1)
        pred_train = np.argmax(gcn(g)[g.train_mask].cpu().numpy(), axis=1)
        acc = accuracy_score(g.y.cpu()[g.test_mask].detach(), predictions)
        acc_train = accuracy_score(g.y.cpu()[g.train_mask].detach(), pred_train)
        print(f"[{epoch + 1:{length}}] loss: {loss.item(): .3f}, "
              f"training accuracy: {acc_train: .3f}, val_accuracy: {acc: .3f}")
    history.append((loss.item(), acc))
    if epoch > 5 and history[-5][0] < history[-1][0] and EARLY_STOPPING:
        print("early stopping activated!")
        break

print("Optimization finished!")
print("Saving model...")
th.save(gcn, f"models/gcn_{datetime.now()}.nn")
with th.no_grad():
    predictions = np.argmax(gcn(g)[g.test_mask].cpu().detach().numpy(), axis=1)
    acc = accuracy_score(g.y.cpu()[g.test_mask].detach(), predictions)
    f1 = f1_score(g.y.cpu()[g.test_mask].detach(), predictions, average='macro')
    conf_mat = confusion_matrix(g.y.cpu()[g.test_mask].detach(), predictions)

print(f"Test Accuracy: {acc: .3f}")
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
