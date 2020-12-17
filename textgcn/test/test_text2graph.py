from unittest import TestCase
import torch as th
from sklearn.feature_extraction.text import CountVectorizer

from textgcn import Text2GraphTransformer


class TestText2Graph(TestCase):
    def test_fit_transform(self):
        inp = ["Far Out in the uncharted backwaters of the unfashionable end of the Western Spiral arm of the galaxy",
               " lies a small unregarded yellow sun."]
