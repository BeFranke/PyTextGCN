import glob
import os
import pickle
import time
from typing import Union, List, Dict

import joblib as jl
import numpy as np
import torch as th
import torch_geometric as tg
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tqdm.asyncio import tqdm
from scipy import sparse as sp

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
                 n_jobs: int = 1, max_df=1.0, verbose=0, rm_stopwords=True, sparse_features=True):
        """
        sklearn-module that transforms a text corpus into a graph, according to the algorithm specified by
        Yao et al, Graph Convolutional Networks for Text Classification (2018), https://arxiv.org/abs/1809.05679.
        :param min_df: Minimum word frequency for the word to be included, can be float (relative frequency)
                        or int (absolute frequency)
        :param window_size: Size of the sliding window. Bigger values mean more edges in the graph.
        :param save_path: Path to save the graph to, optional
        :param n_jobs: number of parallel workers. Currently only used for text preprocessing
        :param max_df: maximum word frequency, similar to min_df
        :param verbose: [0, 1, 2], higher means more debug output
        :param rm_stopwords: weather to remove common words that do not contain much information from the text.
                            currently, nltk is used for this.
        :param sparse_features: if True, feature matrix is computed as sparse. This can save a lot of RAM and GPU memory
                                during training but can only be used if the neural net can handle sparse matrices
        """
        self.sparse_features = sparse_features
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

    def fit_transform(self, X: Union[List[str], str],
                      y: Union[th.Tensor, np.ndarray, List[int], None] = None,
                      test_idx: Union[th.Tensor, np.ndarray, List[int], None] = None) -> tg.data.Data:
        """
        transform input corpus into a torch_geometric Data-object (a graph)
        :param X: corpus, can either be:
                    - List[str], the each list entry is taken as a document (recommended input format)
                    - str, then the string is interpreted as a path towards a folder, all .txt-files found inside the
                      folder are taken as documents
        :param y: list of labels, shape (len(x),)
        :param test_idx: this parameter can tell the downstream neural net which nodes should be used for testing
        :return: the resulting graph as tg.Data object with attributes:
                - x: Node features, shape (n_nodes_, n_nodes_) (MAY BE SPARSE!)
                - y: Node labels, shape (n_nodes_,)
                - edge_index: Adjacency in COO format, shape (2, n_edges_)
                - edge_attr: Edge weights, shape (n_edges,)
                - test_mask: node bitmap showing which (document-)nodes should be used for testing
                - train_idx: node bitmap showing which (document-)nodes should be used for training
                - n_vocab: number of unique words in the vocabulary (also, the lowest document-node index)
        """
        th.set_grad_enabled(False)
        if not isinstance(test_idx, th.LongTensor):
            test_idx = th.LongTensor(test_idx)
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
        self.n_nodes_ = n_docs + n_vocabs
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
        # DONE by making this sparse we could save 2 GB of RAM
        # node_feats = th.eye(n_docs + n_vocabs)
        node_feats = self.node_feats() if self.sparse_features else th.eye(self.n_nodes_)
        test_mask = th.zeros(self.n_nodes_, dtype=th.bool)
        test_mask[test_idx + self.n_vocabs_] = 1
        train_mask = th.logical_not(test_mask)
        train_mask[:self.n_vocabs_] = 0
        # add pseudo-labels for word nodes, so that masks can be directly applied
        y_nodes = th.zeros(self.n_nodes_, dtype=th.long)
        y_nodes[self.n_vocabs_:] = y
        g = tg.data.Data(x=node_feats.float(), edge_index=coo.T, edge_attr=edge_weights.float(), y=y_nodes,
                         test_mask=test_mask, train_mask=train_mask, n_vocab=n_vocabs)

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
        """
        loads a graph from file
        :param save_path: path to file
        :return: graph as in fit_transform
        """
        if not os.path.exists(save_path):
            raise FileNotFoundError("Given file does not exist!")
        with open(save_path, "rb") as fp:
            g = pickle.load(fp)
        return g

    @property
    def vocabulary(self) -> Dict[str, int]:
        """
        :return: mapping of word to vocabulary-index
        """
        return self.cv.vocabulary_

    def node_feats(self):
        """
        computes sparse feature matrix
        :return sparse feature matrix
        """
        # inspired by:
        # https://kenqgu.com/classifying-asian-prejudice-in-tweets-during-covid-19-using-graph-convolutional-networks/
        identity = sp.identity(self.n_nodes_)
        ind0, ind1, vals = sp.find(identity)
        inds = th.stack((th.from_numpy(ind0), th.from_numpy(ind1)))
        return th.sparse_coo_tensor(inds, vals, device=th.device("cpu"), dtype=th.float)
