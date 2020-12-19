import glob
import os
from typing import Union

import joblib as jl
import torch as th
import torch_geometric as tg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from textgcn.lib.pmi import pmi


class Text2GraphTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, word_threshold: Union[int, float] = 5, window_size: int = 15, save_path: str = None,
                 n_jobs: int = 1):
        self.n_jobs = n_jobs
        # assert isinstance(stopwords, list) or stopwords in self.valid_stopwords
        assert word_threshold > 0
        self.word_threshold = word_threshold
        self.save_path = save_path
        self.input = None
        self.cv = None
        self.window_size = window_size

    def fit_transform(self, X, y=None, test_idx=None, **fit_params):
        # load the text
        if isinstance(X, list):
            self.input = X
        else:
            self.input = []
            for f in glob.glob(os.path.join(X, "*.txt")):
                with open(f, 'r') as fp:
                    self.input.append(fp.read())
        # pre-process the text
        self.cv = CountVectorizer(stop_words='english', min_df=self.word_threshold)
        occurrence_mat = self.cv.fit_transform(self.input).toarray()
        # build the graph
        # id-matrix of size n_vocab + n_docs
        n_docs, n_vocabs = occurrence_mat.shape
        node_feats = th.eye(n_docs + n_vocabs)
        # memory-intensive solution: compute PMI and TFIDF matrices and store them
        tfidf_mat = TfidfTransformer().fit_transform(occurrence_mat)
        # pmi_mat = self.pmi_matrix(n_docs, n_vocabs)
        pmi_mat = pmi(self.cv, X, self.window_size, 1, self.n_jobs)

        # build word-document edges. The first value is increased by n_vocab, as documents start at index n_vocab
        docu_coo = th.nonzero(th.from_numpy(occurrence_mat)) + th.Tensor([n_vocabs, 0])
        # build word-word edges
        word_coo = th.nonzero(pmi_mat)
        coo = th.vstack([word_coo, docu_coo])
        edge_weights = th.vstack([tfidf_mat[tuple(docu_coo.T)], pmi_mat[tuple(word_coo.T)]])
        g = tg.data.Data(x=node_feats, edge_index=coo.T, edge_attr=edge_weights, y=y,
                         test_idx=n_vocabs + test_idx,
                         train_idx=[n_vocabs + i for i in range(n_docs) if i not in test_idx])

        return g
