from unittest import TestCase
import torch as th
from sklearn.feature_extraction.text import CountVectorizer

from textgcn import Text2GraphTransformer


class TestText2Graph(TestCase):
    def test_pmi_and_edges(self):
        # simulate 2 documents and a vocabulary of 2 words
        X = th.Tensor([
            [
                [1, 0, 0, 0, 0, 1, 0],
                [0, 1, 1, 0, 0, 0, 0]
            ],
            [
                [0, 1, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 1, 0]
            ]
        ])
        t2t = Text2GraphTransformer(window_size=4)
        t2t.n_vocabs_ = 2
        t2t.n_docs_ = 2
        weights, edges = t2t.pmi_and_edges(X)
        print(edges)
        print(weights)
