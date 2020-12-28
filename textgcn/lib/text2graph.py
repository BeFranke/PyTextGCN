import glob
import itertools
import os
from typing import Union, Dict, List, Tuple
import pickle
import joblib as jl
import torch as th
import torch_geometric as tg
from nltk import RegexpTokenizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import time


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
        # build the graph
        # memory-intensive solution: compute PMI and TFIDF matrices and store them
        tfidf_mat = th.from_numpy(TfidfTransformer().fit_transform(occurrence_mat).todense())

        # build word-document edges. The first value is increased by n_vocab, as documents start at index n_vocab
        docu_coo = th.nonzero(th.from_numpy(occurrence_mat))

        # pmi_mat = self.pmi_matrix(n_docs, n_vocabs)
        edge_ww_weights, edges_coo = self.process_batches(X)

        edge_weights = th.cat([
            tfidf_mat[tuple(docu_coo.T)],
            edge_ww_weights
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
        max_sent_len = max(map(len, X))
        enc = jl.Parallel(n_jobs=self.n_jobs)(
            jl.delayed(self.encode_sentence)(self.n_vocabs_, sent, self.cv.vocabulary_, max_sent_len) for sent in X
        )

        return th.stack(enc)

    def encode_sentence(self, n_vocabs, sent, mapping: Dict[str, int], max_sent_len):
        enc = th.zeros(n_vocabs, max_sent_len)
        for i, tok in enumerate(sent):
            enc[mapping[tok], i] += 1

        return enc

    def pmi_and_edges(self, X: th.Tensor) -> Tuple[int, th.Tensor, th.Tensor]:
        """
        calculates pmi scores and edges
        :param X: 3d tensor of shape (
        :return:
        """
        # set device
        device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        # conv synonym
        conv = th.nn.functional.conv1d
        # dispatch input tensor to GPU if available
        inp = X.to(device)
        # regular summation means all conv-weights are 1
        weights = th.ones((self.n_vocabs_, 1, self.window_size)).to(device)
        # convolution as window-summation
        res: th.Tensor = conv(inp, weights, groups=self.n_vocabs_)
        # binarize result and cast to float, because mean only works on float tensors
        res = (res > 0).float().cpu()
        n_windows = res.shape[1]
        # get rid of minibatch dimension (a.k.a document dimension)
        res = th.cat(tuple(res), dim=1)
        # we now have
        p_i = th.sum(res, dim=1)

        # idx = th.cartesian_prod(th.arange(self.n_vocabs_, dtype=th.long), th.arange(self.n_vocabs_, dtype=th.long))
        # p_ij contains how often a pair of words occured in the same window, indexed by idx
        # p_ij = th.sum(th.prod(res[idx, :], dim=1), dim=1)
        # p_ij = th.zeros(self.n_vocabs_ ** 2)
        # i = 0
        # for id in idx:
        # for id in itertools.product(range(self.n_vocabs_), range(self.n_vocabs_)):
        #     print(f"\r{i:10} of {self.n_vocabs_ ** 2}", end="")
        #     p_ij[i] += th.sum(th.prod(res[id, :], dim=0), dim=0)
        #     i += 1
        # print()
        p_ij = th.FloatTensor(jl.Parallel(n_jobs=self.n_jobs)(
            jl.delayed(
                lambda id: th.sum(th.prod(res[id, :], dim=0), dim=0)
            )(id) for id in itertools.product(range(self.n_vocabs_), range(self.n_vocabs_))
        ))

        # p_ij = th.log(p_ij / th.prod(p_i[idx], dim=1))
        # idx_nonzero = th.nonzero(p_ij > 0)

        return n_windows, p_i, p_ij

    def process_batch(self, X):
        X = self.encode_input(X)
        return self.pmi_and_edges(X)

    def process_batches(self, X):
        i = 0
        wds = 0
        p_is = th.zeros(self.n_vocabs_)
        p_ijs = th.zeros(self.n_vocabs_ ** 2)
        while i < len(X):
            print(f"Processing batch {int(i / self.batch_size + 1)} of {int(len(X) / self.batch_size + 1)}")
            j = min(i + self.batch_size, len(X))
            n_windows, p_i, p_ij = self.process_batch(X[i:j])
            wds += n_windows
            p_is += p_i
            p_ijs += p_ij
            i = j

        p_i = sum(p_is) / sum(wds)
        p_ij = sum(p_ijs) / sum(wds)
        idx = th.cartesian_prod(th.arange(self.n_vocabs_, dtype=th.long), th.arange(self.n_vocabs_, dtype=th.long))
        p_ij = th.log(p_ij / th.prod(p_i[idx], dim=1))
        idx_nonzero = th.nonzero(p_ij > 0)
        return p_ij[idx_nonzero], idx[idx_nonzero]
