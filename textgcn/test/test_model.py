from unittest import TestCase

from sklearn.metrics import accuracy_score
from torch_geometric.datasets import KarateClub
from textgcn.lib.models import GCN
import torch as th
import numpy as np

class TestModel(TestCase):
    def test_GCN(self):
        g = KarateClub().data

        gcn = GCN(g.x.shape[1], len(np.unique(g.y)), n_hidden_gcn=64)

        epochs = 100
        criterion = th.nn.CrossEntropyLoss(reduction='mean')
        optimizer = th.optim.Adam(gcn.parameters(), lr=0.02)

        device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        gcn = gcn.to(device).float()
        g = g.to(device)

        length = len(str(epochs))
        print("#### TRAINING START ####")
        test_mask = th.logical_not(g.train_mask)
        for epoch in range(epochs):
            gcn.train()
            outputs = gcn(g)[g.train_mask]
            loss = criterion(outputs, g.y[g.train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            gcn.eval()
            with th.no_grad():
                predictions = np.argmax(gcn(g)[test_mask].cpu().detach().numpy(), axis=1)
                pred_train = np.argmax(gcn(g)[g.train_mask].cpu().detach().numpy(), axis=1)
                acc = accuracy_score(g.y.cpu()[test_mask].detach(), predictions)
                acc_train = accuracy_score(g.y.cpu()[g.train_mask].detach(), pred_train)
                print(f"[{epoch + 1:{length}}] loss: {loss.item(): .3f}, "
                      f"training accuracy: {acc_train: .3f}, val_accuracy: {acc: .3f}")