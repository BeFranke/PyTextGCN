from unittest import TestCase
import torch as th
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from ..lib.pmi import pmi_document, pmi


class Test(TestCase):
    def test_pmi(self):
        # some nonsense input that does not appear in STOPWORDS
        inp = ["Alpha Beta Gamma Delta Epsilon", "Alpha Beta Mu Epsilon", "Gamma Gamma Delta"]
        cv = CountVectorizer(stop_words='english', min_df=1).fit(inp)
        actual = pmi(cv, inp, 15, 1, 4)
        at = lambda w1, w2: (cv.vocabulary_[w1], cv.vocabulary_[w2])
        self.assertAlmostEqual(np.log(1.5), actual[at('alpha', 'beta')].item(), delta=1e-6)
        self.assertAlmostEqual(np.log(1.5), actual[at('alpha', 'epsilon')].item(), delta=1e-6)
        self.assertAlmostEqual(np.log(1.5), actual[at('gamma', 'delta')].item(), delta=1e-6)

    def test_pmi_document(self):
        inp = ["Far Out in the uncharted backwaters of the unfashionable end of the Western Spiral arm of the galaxy"
               " lies a small unregarded yellow sun."]
        cv = CountVectorizer(stop_words='english', min_df=1).fit(inp)
        p_i, p_ij, n_windows = pmi_document(cv, inp[0], 15, 1)
        self.assertEqual(n_windows, 1)
        self.assertTrue(th.equal(p_i, th.ones(14, dtype=th.long)))
        self.assertTrue(th.equal(p_ij, th.ones((14, 14), dtype=th.long)))

    def test_pmi_2(self):
        # some nonsense input that does not appear in STOPWORDS
        inp = ["Alpha Beta Ceta Delta", "Alpha Ceta",  "Alpha Delta Ceta", "Beta Delta Beta"]
        cv = CountVectorizer(min_df=1).fit(inp)
        mat = pmi(cv, inp, 15, 1, 4)
        at = lambda i, j: (cv.vocabulary_[i], cv.vocabulary_[j])
        print(mat[at('alpha', 'ceta')])