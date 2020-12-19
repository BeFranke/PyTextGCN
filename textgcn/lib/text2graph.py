import glob
import os
from typing import Union
import pickle
import joblib as jl
import torch as th
import torch_geometric as tg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import time

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
        if not isinstance(test_idx, th.Tensor):
            test_idx = th.Tensor(test_idx)
        if y is not None and not isinstance(y, th.LongTensor):
            y = th.LongTensor(y)
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
        tfidf_mat = th.from_numpy(TfidfTransformer().fit_transform(occurrence_mat).todense())
        # pmi_mat = self.pmi_matrix(n_docs, n_vocabs)
        pmi_mat = pmi(self.cv, X, self.window_size, 1, self.n_jobs)

        # build word-document edges. The first value is increased by n_vocab, as documents start at index n_vocab
        docu_coo = th.nonzero(th.from_numpy(occurrence_mat))
        # build word-word edges
        word_coo = th.nonzero(pmi_mat)
        edge_weights = th.cat([
            tfidf_mat[tuple(docu_coo.T)],
            pmi_mat[tuple(word_coo.T)]
        ])
        coo = th.vstack([word_coo, docu_coo + th.Tensor([n_vocabs, 0])]).long()
        g = tg.data.Data(x=node_feats.float(), edge_index=coo.T, edge_attr=edge_weights.float(), y=y,
                         test_idx=(n_vocabs + test_idx).long(),
                         train_idx=th.LongTensor([n_vocabs + i for i in range(n_docs) if i not in test_idx]),
                         n_vocab = n_vocabs)

        if self.save_path is not None:
            print(f"saving to  {self.save_path}")
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            savefile = os.path.join(self.save_path, f"TGData_{time.time()}.p")
            with open(savefile, 'wb') as fp:
                pickle.dump(g, fp)
            print("save successful!")

        return g

    @staticmethod
    def load_graph(save_path):
        if not os.path.exists(save_path):
            raise FileNotFoundError("Given file does not exist!")
        with open(save_path, "rb") as fp:
            g = pickle.load(fp)
        return g
