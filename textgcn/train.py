from textgcn.lib.text2graph import Text2GraphTransformer
import pandas as pd
import numpy as np

train = pd.read_csv("../data/amazon/train_40k.csv")

x_train = train['Text'].tolist()
y_train = train['Cat1'].tolist()

split_idx = int(0.8 * len(x_train))
test_idx = range(split_idx, len(x_train))

t2g = Text2GraphTransformer(n_jobs=1).fit_transform(x_train, y_train, test_idx=test_idx)

