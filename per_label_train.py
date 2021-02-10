from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from textgcn import Text2GraphTransformer
from textgcn.lib.models import *

CPU_ONLY = False
epochs = 500
train_val_split = 0.1
k_split = 3

# lr = 0.2
save_model = False
label_category = "Cat2"
#Hyperparameters to optimize
#learning rate
lr = 0.005
#dropout
do = 0.7
# df_max
df = 0.6

model = GCN

#n_hidden
n_hidden = 100


# resulting dataframe containing all result values
# 2lc = second label classifier
# LR = learning rate
# DO = Drop out rate
# mean acc = Mean accuracy on cross validation scores
# std acc = standart derivation on cross validation scores
# resultDf = pd.DataFrame(index= range(1, int(maxExpSize + 1), 1),
#                        columns= ["2lc", "LR", "DO", "max_df", "model", "mean f1", "std f1"])
# resultDf.fillna(0)


train = pd.read_csv("data/amazon/train.csv")
test = pd.read_csv("data/amazon/test.csv")

save_path = None
x = train['Text'].tolist()
y_top = train['Cat1'].tolist()
y = train['Cat2'].tolist()

x_test = test['Text'].tolist()
y_test = test['Cat2'].tolist()
y_test_top = test['Cat1'].tolist()

val_idx = np.random.choice(len(x), int(train_val_split * len(x)), replace=False)
test_idx = np.arange(len(x), len(x) + len(x_test))

x = x + x_test
y = y + y_test
y_top = y_top + y_test_top

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
scores = []

t2g = Text2GraphTransformer(n_jobs=8, min_df=5, save_path=None, verbose=1, max_df=df)

for classifier in range(num_labels):
    # get index of all labels with category n
    # available_labels = np.nonzero(labels[classifier])[0]
    mask = y_top == classifier
    # build the graph on the basis of the chosen indices
    g = t2g.fit_transform(x, y, test_idx=test_idx, val_idx=val_idx)

    # print(np.unique(g.y[g.test_mask], return_counts=True))
    # print(np.unique(g.y, return_counts=True))
    indices = th.nonzero(th.from_numpy(mask)) + g.n_vocab
    mask = th.zeros(len(g.y), dtype=th.bool)
    mask[indices] = 1


    # relabel the selected labels in ascending order
    le = LabelEncoder()
    newlabels = th.from_numpy(le.fit_transform(th.squeeze(g.y[indices])))[:, None]
    g.y[:] = -1
    g.y[indices] = newlabels
    print(np.unique(g.y[indices]))

    ########################################  define GCN  #####################################
    gcn = model(g.x.shape[1], len(th.unique(g.y[mask])), n_hidden_gcn=n_hidden,
                dropout=do)

    criterion = th.nn.CrossEntropyLoss(reduction='mean')

    device = th.device('cuda' if th.cuda.is_available() and not CPU_ONLY else 'cpu')
    gcn = gcn.to(device).float()
    g = g.to(device)
    mask = mask.to(device)

    # optimizer needs to be constructed AFTER the model was moved to GPU
    optimizer = th.optim.Adam(gcn.parameters(), lr=lr)

    ###############################  Training  ###################################
    length = len(str(epochs))
    time_start = datetime.now()

    train_mask = th.logical_and(g.train_mask, mask)
    val_mask = th.logical_and(g.val_mask, mask)
    test_mask = th.logical_and(g.test_mask, mask)

    for epoch in range(epochs):
        gcn.train()
        outputs = gcn(g)[train_mask]
        loss = criterion(outputs, g.y[train_mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        gcn.eval()
        with th.no_grad():
            logits = gcn(g)
        val_loss = criterion(logits[val_mask], g.y[val_mask])
        pred_val = np.argmax(logits[val_mask].cpu().numpy(), axis=1)
        pred_train = np.argmax(logits[train_mask].cpu().numpy(), axis=1)
        f1_val = f1_score(g.y.cpu()[val_mask], pred_val, average="macro")
        acc_train = accuracy_score(g.y.cpu()[train_mask], pred_train)
        # f1_train = f1_score(g.y.cpu()[train_mask], pred_train)
        print(f"[{epoch + 1:{length}}] loss: {loss.item(): .3f}, "
        f"training accuracy: {acc_train: .3f}, val_f1: {f1_val: .3f}")
    
    scores += [f1_val]
    with open(f"models/amazon/lvl2-cat{classifier}", "wb") as f:
        th.save(gcn, f)

print(scores)

