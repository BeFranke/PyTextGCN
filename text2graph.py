from typing import Union, List
import glob
import os

import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
import torch as th
import joblib as jl
from torch.nn import functional as f


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

    def fit_transform(self, X, y=None, mask=None, **fit_params):
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
        pmi_mat = self.pmi_matrix()

        # build word-document edges. The first value is increased by n_vocab, as documents start at index n_vocab
        docu_coo = th.nonzero(occurrence_mat) + th.Tensor([n_vocabs, 0])
        # build word-word edges
        word_coo = th.vstack(jl.Parallel(n_jobs=self.n_jobs)(
            jl.delayed(self.word_edges_from_doc)(i, x, pmi_mat)
            for i, x in enumerate(occurrence_mat)
        ))
        coo = th.vstack([word_coo, docu_coo])
        g = ...  # set up tg.data object

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
            for i in range(self.window_size):
                idx_1 = self.cv.vocabulary_[doc_words[window_start + i]]
                freq_singular[dx_1] += 1
                for j in range(self.window_size - i):
                    idx_2 = self.cv.vocabulary_[doc_words[window_start + i + j]]
                    freq_dual[idx_1][idx_2] += 1

            n_windows += 1

        return freq_singular, freq_dual, n_windows

    def word_edges_from_doc(i, x, pmi_mat):
        occ = th.nonzero(x) * pmi_mat


def pmi(cv: CountVectorizer, documents, window_size, strides):
    vocab_size = len(cv.vocabulary_.values())

    p_i = th.zeros(vocab_size)
    p_ij = th.zeros((vocab_size, vocab_size))
    total_windows = 0
    num_documents = len(documents)
    # todo: parallelize
    for i, document in enumerate(documents):
        # print(document)
        result = pmi_document(cv, document, window_size, strides)
        p_i = p_i + result[0]
        p_ij = p_ij + result[1]
        total_windows = total_windows + result[2]
        if i % 1000 == 0:
            print(f"Processed {i} of {num_documents} documents.")
    # normalization:
    p_i = p_i / total_windows
    p_ij = p_ij / total_windows

    pm_ij = th.log(th.divide(p_ij, th.outer(p_i, p_i)))  # outer product to get every ij combination
    return pm_ij


def pmi_document(cv, document, window_size, strides):
    # sample sentence:
    # cv = CountVectorizer(stop_words='english', min_df=1)
    # cv.fit(corpus)

    # encode each word individually to get one-hot encoding
    encoded_sentence = cv.transform(document.split()).todense()
    if encoded_sentence.shape[0] <= 1 or encoded_sentence.shape[0] < window_size:  # todo implement padding?
        return 0, 0, 0
    t = th.tensor(encoded_sentence)

    # sliding window over one-hot encoding of sentence
    sliding_window = t.unfold(dimension=0, size=window_size, step=strides)

    # total number of sliding windows:
    num_windows = sliding_window.shape[0]
    #  print(f"Windows: {num_windows}")
    vocab_size = sliding_window.shape[1]
    # sum one-hot encodings over all words and windows => number of occurrences per token in vocabulary
    # = number of sliding windows that contains the token:
    p_i = sliding_window.sum(dim=(0, 2))

    # reduce each window to an encoding indication which tokens occur in the window
    occurrences = th.min(sliding_window.sum(dim=2), th.tensor(1))
    # sum of outer product of tokens
    # => occurence matrix (except diagonal, which is increased for each occurrence of the token)
    p_ij = th.einsum('ij,ik->jk', occurrences, occurrences)
    return p_i, p_ij, num_windows  # we need to accummulate those for each sentence
    # note: log(0) = -inf


def pmi_test():
    import pandas as pd

    path = "text_gcn/data/amazon"
    data = pd.read_csv(path + "/train.csv")
    data = data[['Text']]  # only extrat text
    data = data.dropna()

    cv = CountVectorizer(stop_words='english', min_df=5, max_features=300)
    cv.fit(data["Text"])

    result = pmi(cv, data["Text"], window_size=20, strides=1)
    print(result)


pmi_test()
