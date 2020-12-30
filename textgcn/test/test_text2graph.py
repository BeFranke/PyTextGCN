from unittest import TestCase
import torch as th
from sklearn.feature_extraction.text import CountVectorizer

from textgcn import Text2GraphTransformer


class TestText2Graph(TestCase):
    def test_fit(self):
        t2g = Text2GraphTransformer(word_threshold=1, window_size=3)


