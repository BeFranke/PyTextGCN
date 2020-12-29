import glob
import itertools
import os
from typing import Union, Dict, List, Tuple
import pickle
import joblib as jl
import torch as th
import numpy as np
import torch_geometric as tg
from nltk import RegexpTokenizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import time
from clib.graphbuilder import compute_word_word_edges


class Text2GraphTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, word_threshold: Union[int, float] = 5, window_size: int = 15, save_path: str = None,
                 n_jobs: int = 1, max_df=0.9, batch_size=400):
        self.batch_size = batch_size
        self.max_df = max_df
        self.n_jobs = n_jobs
        # assert isinstance(stopwords, list) or stopwords in self.valid_stopwords
        assert word_threshold > 0
        self.word_threshold = word_threshold
        self.save_path = save_path
        self.input = None
        self.cv = None
        self.window_size = window_size

    def fit_transform(self, X, y=None, test_idx=None, **fit_params):
        th.set_grad_enabled(False)
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
        self.cv = CountVectorizer(stop_words='english', min_df=self.word_threshold, max_df=self.max_df)
        occurrence_mat = self.cv.fit_transform(self.input).toarray()
        n_docs, n_vocabs = occurrence_mat.shape
        self.n_docs_ = n_docs
        self.n_vocabs_ = n_vocabs
        X = self.encode_input(X)
        # build the graph
        # memory-intensive solution: compute PMI and TFIDF matrices and store them
        tfidf_mat = th.from_numpy(TfidfTransformer().fit_transform(occurrence_mat).todense())

        # build word-document edges. The first value is increased by n_vocab, as documents start at index n_vocab
        docu_coo = th.nonzero(th.from_numpy(occurrence_mat))

        # pmi_mat = self.pmi_matrix(n_docs, n_vocabs)
        edges_coo, edge_ww_weights = compute_word_word_edges(X, self.n_vocabs_, self.n_docs_, self.max_sent_len_,
                                                             self.window_size, self.n_jobs, self.batch_size)

        edges_coo, edge_ww_weights = th.from_numpy(edges_coo), th.from_numpy(edge_ww_weights)

        edge_weights = th.cat([
            edge_ww_weights,
            tfidf_mat[tuple(docu_coo.T)]
        ])
        coo = th.vstack([edges_coo, docu_coo + th.Tensor([n_vocabs, 0])]).long()
        # id-matrix of size n_vocab + n_docs
        node_feats = th.eye(n_docs + n_vocabs)
        g = tg.data.Data(x=node_feats.float(), edge_index=coo.T, edge_attr=edge_weights.float(), y=y,
                         test_idx=(n_vocabs + test_idx).long(),
                         train_idx=th.LongTensor([n_vocabs + i for i in range(n_docs) if i not in test_idx]),
                         n_vocab=n_vocabs)

        if self.save_path is not None:
            print(f"saving to  {self.save_path}")
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            savefile = os.path.join(self.save_path, f"TGData_{time.time()}.p")
            with open(savefile, 'wb') as fp:
                pickle.dump(g, fp)
            print("save successful!")
        th.set_grad_enabled(True)
        return g

    @staticmethod
    def load_graph(save_path):
        if not os.path.exists(save_path):
            raise FileNotFoundError("Given file does not exist!")
        with open(save_path, "rb") as fp:
            g = pickle.load(fp)
        return g

    def encode_input(self, X):
        X = jl.Parallel(n_jobs=self.n_jobs)(
            jl.delayed(
                lambda doc: [
                    x.lower() for x in RegexpTokenizer(r"\w+").tokenize(doc) if x.lower() in self.cv.vocabulary_
                ]
            )(doc) for doc in X
        )
        self.max_sent_len_ = max(map(len, X))
        enc = np.stack(jl.Parallel(n_jobs=self.n_jobs)(
            jl.delayed(self.encode_sentence)(self.n_vocabs_, sent, self.cv.vocabulary_, self.max_sent_len) for sent in X
        ))

        return enc

    def encode_sentence(self, n_vocabs, sent, mapping: Dict[str, int], max_sent_len):
        enc = np.empty(n_vocabs, max_sent_len, dtype=np.int32)
        enc[:] = -1
        for i, tok in enumerate(sent):
            enc[i] = mapping[tok]

        return enc
