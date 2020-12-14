from unittest import TestCase

from sklearn.feature_extraction.text import CountVectorizer

from textgcn.lib.pmi import pmi_document


class Test(TestCase):
    def test_pmi(self):
        self.fail()

    def test_pmi_document(self):
        inp = ["Far Out in the uncharted backwaters of the unfashionable end of the Western Spiral arm of the galaxy"
               " lies a small unregarded yellow sun."]
        cv = CountVectorizer(stop_words='english', min_df=1).fit(inp)
        result = pmi_document(cv, inp[0], 15, 1)
