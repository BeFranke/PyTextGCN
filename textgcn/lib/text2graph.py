import glob
import os
import pickle
import time
from typing import Union

import joblib as jl
import numpy as np
import torch as th
import torch_geometric as tg
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tqdm.asyncio import tqdm

from .clib.graphbuilder import compute_word_word_edges


def _encode_input(X, n_jobs, vocabulary, verbose, n_docs):
    if verbose > 0:
        print("Tokenizing text and removing unwanted words...")
    X = jl.Parallel(n_jobs=n_jobs)(
        jl.delayed(
            lambda doc: [
                x.lower() for x in nltk.RegexpTokenizer(r"\w+").tokenize(doc) if x.lower() in vocabulary
            ]
        )(doc) for doc in tqdm(X)
    )
    max_sent_len = max(map(len, X))
    if verbose > 1:
        print(f"Sequence length is {max_sent_len}")
    if verbose > 0:
        print("Padding and encoding text...")
    X = np.array(jl.Parallel(n_jobs=8)(
        jl.delayed(
            lambda doc: [vocabulary[w] for w in doc] + [-1] * (max_sent_len - len(doc))
        )(doc) for doc in tqdm(X)
    ), dtype=np.int32)
    assert X.shape == (n_docs, max_sent_len)
    return X, max_sent_len


class Text2GraphTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, min_df: Union[int, float] = 5, window_size: int = 20, save_path: str = None,
                 n_jobs: int = 1, max_df=1.0, verbose=0, rm_stopwords=True):
        self.rm_stopwords = rm_stopwords
        self.verbose = verbose
        self.max_df = max_df
        self.n_jobs = n_jobs
        # assert isinstance(stopwords, list) or stopwords in self.valid_stopwords
        assert min_df > 0
        self.min_df = min_df
        self.save_path = save_path
        self.input = None
        self.cv = None
        self.window_size = window_size
        self.stop_words = None
        if self.rm_stopwords:
            nltk.download('stopwords')
            self.stop_words = set(nltk.corpus.stopwords.words('english'))

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
            if self.verbose > 0:
                print(f"Loading input from {X}")
            self.input = []
            for f in glob.glob(os.path.join(X, "*.txt")):
                with open(f, 'r') as fp:
                    self.input.append(fp.read())
        # pre-process the text
        self.cv = CountVectorizer(stop_words=self.stop_words, min_df=self.min_df, max_df=self.max_df)
        occurrence_mat = self.cv.fit_transform(self.input).toarray()
        n_docs, n_vocabs = occurrence_mat.shape
        if self.verbose > 1:
            print(f"Number of documents in input: {n_docs}")
            print(f"Vocabulary size: {n_vocabs}")
        self.n_docs_ = n_docs
        self.n_vocabs_ = n_vocabs
        X, self.max_sent_len_ = _encode_input(X, self.n_jobs, self.cv.vocabulary_, self.verbose, self.n_docs_)
        # build the graph
        if self.verbose > 0:
            print(f"Building doc-word edges...")

        tfidf_mat = th.from_numpy(TfidfTransformer().fit_transform(occurrence_mat).todense())

        # build word-document edges
        docu_coo = th.nonzero(th.from_numpy(occurrence_mat))
        # edge from i to j also means edge from j to i
        docu_coo_sym = th.flip(docu_coo, dims=[1])

        if self.verbose > 0:
            print(f"building word-word.edges...")

        # call compute_word_word_edges and map the two resulting numpy arrays to torch
        edges_coo, edge_ww_weights = map(
            th.from_numpy,
            compute_word_word_edges(X, self.n_vocabs_, self.n_docs_, self.max_sent_len_, self.window_size,
                                    self.n_jobs, self.verbose)
        )

        edge_weights = th.cat([
            edge_ww_weights,
            tfidf_mat[tuple(docu_coo.T)],
            tfidf_mat[tuple(docu_coo.T)]
        ])
        coo = th.vstack([
            edges_coo,
            docu_coo + th.Tensor([n_vocabs, 0]),
            docu_coo_sym + th.Tensor([0, n_vocabs])
        ]).long()

        if self.verbose > 0:
            print(f"total edge shape is {coo.shape}")

        # id-matrix of size n_vocab + n_docs
        # TODO by making this sparse we could save 2 GB of RAM
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

    @property
    def vocabulary(self):
        return self.cv.vocabulary_
