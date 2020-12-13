from unittest import TestCase
import torch as th

from textgcn import Text2Graph


class TestText2Graph(TestCase):
    def test_fit_transform(self):
        self.fail()

    def test_pmi_matrix(self):
        self.fail()

    def test_pmi_from_doc(self):
        self.fail()

    def test_word_edges_from_doc(self):
        pmi_mat = th.Tensor([[1, 2, 3, 0], [5, 1, 7, 8], [9, 10, 1, 12], [13, 14, 15, 1]])
        x = th.Tensor([1, 0, 1, 1])

        expected = th.Tensor([[0, 0], [0, 2], [2, 0], [2, 2], [2, 3], [3, 0], [3, 2], [3, 3]])
        actual = Text2Graph.word_edges_from_doc(x, pmi_mat)
        self.assertTrue(th.equal(expected, actual))
