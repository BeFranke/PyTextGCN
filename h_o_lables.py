from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from textgcn import Text2GraphTransformer
from textgcn.lib.models import *

CPU_ONLY = False
epochs = 100
train_val_split = 0.1
k_split = 5

# lr = 0.2
save_model = False
label_category = "Cat2"
#Hyperparameters to optimize
#learning rate
lrs = [0.0001, 0.001, 0.01, 0.1]
#dropout
dos = [0.5, 0.7]
# df_max
dfs = [0.5, 0.6, 0.7]

models = [GCN, EGCN]

#n_hidden
n_hidden = 100
frameIterator = 0
maxExpSize = len(lrs) * len(dos) * len(dfs) * len(models)

# resulting dataframe containing all result values
# 2lc = second label classifier
# LR = learning rate
# DO = Drop out rate
# mean acc = Mean accuracy on cross validation scores
# std acc = standart derivation on cross validation scores
resultDf = pd.DataFrame(index= range(1, int(maxExpSize + 1), 1),
                        columns= ["2lc", "LR", "DO", "max_df", "model", "mean f1", "std f1"])
resultDf.fillna(0)


train = pd.read_csv("data/amazon/train.csv")

#save_path = "textgcn/graphs/"
save_path = None
x = train['Text'].tolist()
y_top = train['Cat1'].tolist()
y = train['Cat2'].tolist()

y_top = LabelEncoder().fit_transform(y_top)
y = LabelEncoder().fit_transform(y)

num_labels = len(np.unique(y_top))

# separated labels variable into 6 bitmap vectors
labels = np.zeros((num_labels, len(y_top)))
for i in np.arange(len(y_top)):
    labels[y_top[i]][i] = 1

# Train/val split
#test_idx = np.random.choice(len(x), int(train_val_split * len(x)), replace=False)

print("Data loaded!")
timestamp = datetime.now().strftime("%d_%b_%y_%H_%M_%S")
csv_name = "HypOpt_Labels_" + label_category + "_" + timestamp + ".csv"

################################################  Text to Graph ################################################
for mdf in dfs:
    t2g = Text2GraphTransformer(n_jobs=8, min_df=5, save_path=save_path, verbose=1, max_df=mdf)

    for classifier in range(num_labels):
        # get index of all labels with category n
        available_labels = np.nonzero(labels[classifier])[0]
        mask = y_top == classifier
        # build the graph on the basis of the chosen indices
        g = t2g.fit_transform(x, y, test_idx=[])

        kf = KFold(n_splits=k_split, shuffle=True)

        print(np.unique(g.y[g.test_mask], return_counts=True))
        print(np.unique(g.y, return_counts=True))
        indices = th.nonzero(th.from_numpy(mask)) + g.n_vocab


        # relabel the selected labels in ascending order
        g.y[indices] = th.from_numpy(LabelEncoder().fit_transform(th.squeeze(g.y[indices])))[:, None]

        for model in models:
            model_name = "GCN" if model == GCN else "EGCN"
            for dropout in dos:
                for lr in lrs:
                    scores = np.zeros(k_split)
                    classifier_name = f"classifier_{classifier}"
                    try:
                        for i, (train, test) in enumerate(kf.split(indices)):
                            train = train + g.n_vocab
                            test = test + g.n_vocab
                            g.test_mask[:] = 0
                            g.test_mask[indices[test]] = 1
                            g.train_mask[:] = 0
                            g.train_mask[indices[train]] = 1
                            ########################################  define GCN  #####################################
                            gcn = model(g.x.shape[1], len(th.unique(g.y[g.train_mask])), n_hidden_gcn=n_hidden,
                                        dropout=dropout)

                            criterion = th.nn.CrossEntropyLoss(reduction='mean')

                            device = th.device('cuda' if th.cuda.is_available() and not CPU_ONLY else 'cpu')
                            gcn = gcn.to(device).float()
                            g = g.to(device)


                            # optimizer needs to be constructed AFTER the model was moved to GPU
                            optimizer = th.optim.Adam(gcn.parameters(), lr=lr)

                            ###############################  Training  ###################################
                            length = len(str(epochs))
                            print("[{}/{}] ----- split {} -- lr: {} -- do: {} - max_df: {}"
                                  .format(frameIterator+1, maxExpSize, i, lr, dropout, mdf))
                            time_start = datetime.now()

                            for epoch in range(epochs):
                                gcn.train()
                                outputs = gcn(g)[g.train_mask]
                                loss = criterion(outputs, g.y[g.train_mask])
                                optimizer.zero_grad(set_to_none=True)
                                loss.backward()
                                optimizer.step()
                                gcn.eval()
                                with th.no_grad():
                                    logits = gcn(g)
                                val_loss = criterion(logits[g.test_mask], g.y[g.test_mask])
                                pred_val = np.argmax(logits[g.test_mask].cpu().numpy(), axis=1)
                                pred_train = np.argmax(logits[g.train_mask].cpu().numpy(), axis=1)
                                f1_val = f1_score(g.y.cpu()[g.test_mask], pred_val, average="macro")
                                acc_train = accuracy_score(g.y.cpu()[g.train_mask], pred_train)
                                print(f"[{epoch + 1:{length}}] loss: {loss.item(): .3f}, "
                                f"training accuracy: {acc_train: .3f}, val_f1: {f1_val: .3f}")
                            scores[i] = f1_val

                        frameIterator += 1
                        # Dataframe structure: "LR", "DO", "n_hidden", "accuracy mean", "accuracy std"
                        resultDf.loc[frameIterator] = [classifier_name, lr, dropout, mdf, model_name,
                                                       scores.mean(), scores.std()]

                        resultDf.to_csv(csv_name, encoding='utf-8')

                    except RuntimeError:
                        print("CUDA ran out of memory. Setting NaN.")
                        resultDf.loc[frameIterator] = [classifier_name, lr, dropout, mdf, model_name, np.nan, np.nan]

print("Optimization finished!")

