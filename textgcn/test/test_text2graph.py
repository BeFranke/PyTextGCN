from unittest import TestCase
import torch as th
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from textgcn import Text2GraphTransformer
import pandas as pd

class TestText2Graph(TestCase):
    def test_inspect_adjacency(self):
        # This "test" is best viewed in PyCharm debugger, put a breakpoint at "print(df)" and look at the
        # resulting adacency-weight matrix. You will see that it makes sense.
        t2g = Text2GraphTransformer(min_df=1, window_size=3, rm_stopwords=True)

        X = ["Time is an illusion. Lunchtime doubly so.",
             "The ships hung in the sky in much the same way that bricks don't.",
             "If there's anything more important than my ego around, I want it caught and shot now.",
             "Would it save you a lot of time if I just gave up and went mad now?"]

        res = t2g.fit_transform(X, y=[1, 0, 1], test_idx=[2])

        # expand adjacency matrix
        A = np.zeros((t2g.n_vocabs_ + 4, t2g.n_vocabs_ + 4))

        for (e1, e2), weight in zip(res.edge_index.T, res.edge_attr):
            A[e1, e2] = weight

        inv_map = {v: k for k, v in t2g.vocabulary.items()}
        inv_map[t2g.n_vocabs_] = "Document 1"
        inv_map[t2g.n_vocabs_ + 1] = "Document 2"
        inv_map[t2g.n_vocabs_ + 2] = "Document 3"
        inv_map[t2g.n_vocabs_ + 3] = "Document 4"
        names = [inv_map[i] for i in range(t2g.n_vocabs_ + 4)]
        df = pd.DataFrame(data=A, columns=names, index=names)
        print(df)