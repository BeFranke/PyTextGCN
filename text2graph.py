from typing import Union, List
import glob
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import torch as th
import joblib as jl


class Text2Graph(BaseEstimator, TransformerMixin):
    valid_stopwords = ['sklearn', 'nltk']

    def __init__(self, word_threshold: int = 5, save_path: str = None, n_jobs: int = 1):
        self.n_jobs = n_jobs
        nltk.download('punkt')
        # assert isinstance(stopwords, list) or stopwords in self.valid_stopwords
        assert word_threshold > 0
        self.word_threshold = word_threshold
        self.save_path = save_path
        self.input = None
        self.cv = None

    def fit_transform(self, X, y=None, mask=None, **fit_params):
        # load the text
        if isinstance(input, list):
            self.input = X
        else:
            self.input = []
            for f in glob.glob(os.path.join(X, "*.txt")):
                with open(f, 'r') as fp:
                    self.input.append(fp.read())
        # pre-process the text
        self.cv = CountVectorizer(stop_words='english', min_df=self.word_threshold)
        occurrence_mat = self.cv.fit_transform(self.input).toarray()
        del self.input
        # build the graph
        # TODO: do NOT forget to call to_undirected at the end of this, else we have an directed graph
        # id-matrix of size n_vocab + n_docs
        n_docs, n_vocabs = occurrence_mat.shape
        node_feats = th.eye(n_docs + n_vocabs)
        # build word-document edges. The first value is increased by n_vocab, as documents start at index n_vocab
        docu_coo = th.nonzero(occurrence_mat) + th.Tensor([n_vocabs, 0])
        # build word-word edges
        word_coo = th.vstack(jl.Parallel(n_jobs=self.n_jobs)(
            jl.delayed(lambda i, x: th.hstack([th.nonzero(x), th.Tensor([[i + n_vocabs]] * len(th.nonzero(x)))]))
            for i, x in enumerate(occurrence_mat)
        ))
        coo = th.vstack([docu_coo, word_coo])
        g = ...         # set up tg.data object
