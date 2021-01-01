from textgcn.lib.text2graph import Text2GraphTransformer
import pandas as pd
import numpy as np
from textgcn.lib.models import GCN
import torch as th
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os
import torch_geometric as tg

CPU_ONLY = False
EARLY_STOPPING = False

train = pd.read_csv("data/amazon/train_40k.csv")
# save_path="textgcn/graphs/"
save_path = None

x = train['Text'].tolist()
y = train['Cat1'].tolist()

y = LabelEncoder().fit_transform(y)

test_idx = np.random.choice(len(x), int(0.1 * len(x)), replace=False)
train_idx = np.array([x for x in range(len(x)) if x not in test_idx])
print("Data loaded!")

t2g = Text2GraphTransformer(n_jobs=8, min_df=5, save_path=save_path, verbose=1, max_df=0.9, sparse_features=False)
ls = os.listdir("textgcn/graphs")
if not ls:
    g = t2g.fit_transform(x, y, test_idx=test_idx)
else:
    g = t2g.load_graph(os.path.join("textgcn/graphs", ls[0]))

print("Graph built")

gcn = GCN(g.x.shape[1], len(np.unique(y)), n_hidden_gcn=200)

epochs = 100
criterion = th.nn.CrossEntropyLoss(reduction='mean')

device = th.device('cuda' if th.cuda.is_available() and not CPU_ONLY else 'cpu')
cpu = th.device('cpu')
print(f"Device: {device}")

# g = g.to(device)
gcn = gcn.float()
loss_history = []
length = len(str(epochs))

# sampler = tg.data.GraphSAINTRandomWalkSampler(data=g, batch_size=1024, walk_length=4, num_steps=epochs, sample_coverage=0)
sampler = tg.data.NeighborSampler
optimizer = th.optim.Adam(gcn.parameters(), lr=0.02)
gcn = gcn.to(device)
print("#### TRAINING START ####")
# for epoch in range(epochs):
for epoch, batch in enumerate(sampler):
    gcn = gcn.to(device)
    batch = batch.to(device)
    gcn.train()
    outputs = gcn(batch)[batch.train_mask]#.to(cpu)
    y_true = batch.y[batch.train_mask]#.to(cpu)
    loss = criterion(outputs, y_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if epoch > 5 and loss_history[-5] < loss_history[-1] and EARLY_STOPPING:
        print("early stopping activated!")
        break
    gcn.eval()
    with th.no_grad():
        gcn = gcn.cpu()
        predictions = np.argmax(gcn(g)[g.test_mask].cpu().numpy(), axis=1)
        pred_train = np.argmax(gcn(g)[g.train_mask].cpu().numpy(), axis=1)
        acc = accuracy_score(g.y.cpu()[g.test_mask].detach(), predictions)
        acc_train = accuracy_score(g.y.cpu()[g.train_mask].detach(), pred_train)
        print(f"[{epoch + 1:{length}}] loss: {loss.item(): .3f}, "
              f"training accuracy: {acc_train: .3f}, val_accuracy: {acc: .3f}")

print("Optimization finished!")
with th.no_grad():
    gcn = gcn.cpu()
    predictions = np.argmax(gcn(g)[g.test_mask].cpu().detach().numpy(), axis=1)
    acc = accuracy_score(g.y.cpu()[g.test_mask].detach(), predictions)
    f1 = f1_score(g.y.cpu()[g.test_mask].detach(), predictions, average='macro')
    conf_mat = confusion_matrix(g.y.cpu()[g.test_mask].detach(), predictions)

print(f"Test Accuracy: {acc: .3f}")
print(f"F1-Macro: {f1: .3f}")
print("Confusion matrix:")
print(conf_mat)