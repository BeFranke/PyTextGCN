from unittest import TestCase
import torch as th
from sklearn.feature_extraction.text import CountVectorizer

from textgcn.lib.pmi import pmi_document


class Test(TestCase):
    def test_pmi(self):
        self.fail()

    def test_pmi_document(self):
        inp = ["Far Out in the uncharted backwaters of the unfashionable end of the Western Spiral arm of the galaxy"
               " lies a small unregarded yellow sun."]
        cv = CountVectorizer(stop_words='english', min_df=1).fit(inp)
        p_i, p_ij, n_windows = pmi_document(cv, inp[0], 15, 1)
        self.assertEqual(n_windows, 1)
        self.assertTrue(th.equal(p_i, th.ones(10, dtype=th.long)))
        self.assertTrue(th.equal(p_ij, th.ones((10, 10), dtype=th.long)))
