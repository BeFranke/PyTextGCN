from textgcn.lib.text2graph import Text2GraphTransformer
import pandas as pd
import numpy as np
from textgcn.lib.models import GCN
import torch as th
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

train = pd.read_csv("../data/amazon/train_40k.csv")

x_train = train['Text'].tolist()
y_train = train['Cat1'].tolist()

y_train = LabelEncoder().fit_transform(y_train)

split_idx = int(0.8 * len(x_train))
test_idx = range(split_idx, len(x_train))

t2g = Text2GraphTransformer(n_jobs=1, word_threshold=5, save_path="./graphs/")
ls = os.listdir("./graphs")
if not ls:
    g = t2g.fit_transform(x_train, y_train, test_idx=test_idx)
else:
    g = t2g.load_graph(os.path.join("./graphs", ls[0]))
gcn = GCN(g.x.shape[1], len(np.unique(y_train)), n_hidden_gcn=300)

epochs = 100
criterion = th.nn.CrossEntropyLoss()
optimizer = th.optim.Adam(gcn.parameters(), lr=0.01)

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
gcn = gcn.to(device).float()
g = g.to(device)

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = gcn(g)[g.train_idx]
    loss = criterion(outputs, g.y[g.train_idx - g.n_vocab])
    loss.backward()
    optimizer.step()
    print(f"[{epoch + 1}] loss: {loss.item(): .3f}")

print("Optimization finished!")
predictions = np.argmax(gcn(g)[g.test_idx].cpu().detach().numpy(), axis=1)
acc = accuracy_score(g.y.cpu()[g.test_idx - g.n_vocab].detach(), predictions)
f1 = f1_score(g.y.cpu()[g.test_idx - g.n_vocab].detach(), predictions, average='macro')

print(f"Test Accuracy: {acc: .3f}")
print(f"F1-Macro: {f1: .3f}")
