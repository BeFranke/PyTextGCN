from unittest import TestCase
import torch as th
from sklearn.feature_extraction.text import CountVectorizer

from textgcn import Text2GraphTransformer


class TestText2Graph(TestCase):
    def test_fit_transform(self):
        inp = ["Far Out in the uncharted backwaters of the unfashionable end of the Western Spiral arm of the galaxy",
               " lies a small unregarded yellow sun."]


    """
    def test_pmi_matrix(self):
        self.fail()

    def test_pmi_from_doc(self):
        inp = ["Far Out in the uncharted backwaters of the unfashionable end of the Western Spiral arm of the galaxy"
               " lies a small unregarded yellow sun."]
        t = Text2Graph()
        t.cv = CountVectorizer(stop_words='english', min_df=0).fit(inp)
        t.input = inp
        actual_sing, actual_dual, actual_n = t.pmi_from_doc(0, len(t.cv.vocabulary_))
        expected_n = 1
        expected_sing = th.ones(14)
        expected_dual = th.ones((14, 14))

        self.assertEqual(actual_n, expected_n)
        self.assertTrue(th.equal(expected_sing, actual_sing))
        self.assertTrue(th.equal(expected_dual, actual_dual))
    """

    def test_word_edges_from_doc(self):
        pmi_mat = th.Tensor([[1, 2, 3, 0], [5, 1, 7, 8], [9, 10, 1, 12], [13, 14, 15, 1]])
        x = th.Tensor([1, 0, 1, 1])

        expected = th.Tensor([[0, 0], [0, 2], [2, 0], [2, 2], [2, 3], [3, 0], [3, 2], [3, 3]])
        actual = Text2GraphTransformer.word_edges_from_doc(x, pmi_mat)
        self.assertTrue(th.equal(expected, actual))
