from typing import Union, List
import glob
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
import torch as th
import joblib as jl


class Text2Graph(BaseEstimator, TransformerMixin):
    valid_stopwords = ['sklearn', 'nltk']

    def __init__(self, word_threshold: int = 5, window_size: int = 15, save_path: str = None, n_jobs: int = 1):
        self.n_jobs = n_jobs
        nltk.download('punkt')
        # assert isinstance(stopwords, list) or stopwords in self.valid_stopwords
        assert word_threshold > 0
        self.word_threshold = word_threshold
        self.save_path = save_path
        self.input = None
        self.cv = None
        self.window_size = window_size

    def fit_transform(self, X, y=None, train_idx=None, test_idx=None, **fit_params):
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
        tfidf_mat = TfidfTransformer().fit_transform(occurance_mat)
        pmi_mat = self.pmi_matrix()

        # build word-document edges. The first value is increased by n_vocab, as documents start at index n_vocab
        docu_coo = th.nonzero(occurrence_mat) + th.Tensor([n_vocabs, 0])
        # build word-word edges
        word_coo = th.vstack(jl.Parallel(n_jobs=self.n_jobs)(
            jl.delayed(self.word_edges_from_doc)(i, x, pmi_mat)
            for i, x in enumerate(occurrence_mat)
        ))
        coo = th.vstack([word_coo, docu_coo])
        edge_weights = th.vstack([tfidf_mat[docu_coo], pmi_mat[word_coo]])
        g = tg.Data(x=edge_feats, edge_index=coo.T, edge_attr=edge_weights, y=y, train_idx=train_idx, test_idx=test_idx)

        return g


    def pmi_matrix(self, n_docs):
        # this is bad and untested, the idea is to compute PMI matrices for each document and then combine them
        freq_singular, freq_dual, n_windows = zip(*jl.Parallel(n_jobs=self.n_jobs)(
                jl.delayed(self.pmi_from_doc)(i) for i in range(n_docs)
        ))
        freq_singular = th.sum(th.stack(th.freq_singular), dim=0)
        freq_dual = th.sum(th.stack(freq_dual), dim=0)
        n_windows = sum(n_windows)
        freq_singular /= n_windows
        freq_dual /= n_windows
        freq_dual = th.log(freq_dual / th.outer(freq_singular, freq_singular))

        return freq_dual


    def pmi_from_doc(self, doc_idx):
        # this is even worse and untested, uses for loops to apply a sliding window over the document (infrequent words are ignored)
        n_windows = 0
        doc_words = self.input[doc_idx].split()
        doc_words = [word for word in doc_words if word in self.cv.vocabulary_]
        freq_singular = th.zeros(n_vocabs)
        freq_dual = th.zeros(n_vocabs * n_vocabs)
        for window_start in range(len(doc_words) - self.window_size + 1):
            for i in range(self.window_size)):
                idx_1 = self.cv.vocabulary_[doc_words[window_start + i]]
                freq_singular[dx_1] += 1
                for j in range(self.window_size - i):
                    idx_2 = self.cv.vocabulary_[doc_words[window_start + i + j]]
                    freq_dual[idx_1][idx_2] += 1

            n_windows += 1

        return freq_singular, freq_dual, n_windows

    def word_edges_from_doc(i, x, pmi_mat):
        # all words that occur in this document
        occ = th.nonzero(x)
        # all combiantions of occ
        edges = th.cartesian_prod(occ, occ)
        # pmi_mat[edges] should return a list of all edge weights in order of the edges, > 0 binarizes this weight
        w = pmi_mat[edges] > 0
        # only take edges with positive weight
        edges = edges[w > 0]
        return edges




