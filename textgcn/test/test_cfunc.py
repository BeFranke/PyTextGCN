from unittest import TestCase
import pandas as pd
from nltk import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm.asyncio import tqdm

from textgcn.lib import sliding_window_tester, test_sym_matrix, compute_word_word_edges
from datetime import datetime
import joblib as jl
import numpy as np


class TestGraphBuilder(TestCase):
    def test_sliding_window_time(self):
        print("loading and parsing data, this might take a few seconds...")
        time = [datetime.now()]
        train = pd.read_csv("../../data/amazon/train_40k.csv")

        X = train['Text'].tolist()
        cv = CountVectorizer(stop_words="english", min_df=5, max_df=0.9).fit(X)
        n_vocab = len(cv.vocabulary_)
        n_documents = len(X)
        X = jl.Parallel(n_jobs=8)(
            jl.delayed(
                lambda doc: [
                    x.lower() for x in RegexpTokenizer(r"\w+").tokenize(doc) if x.lower() in cv.vocabulary_
                ]
            )(doc) for doc in X
        )
        max_sent_len = max(map(len, X))
        X = np.array(jl.Parallel(n_jobs=8)(
            jl.delayed(
                lambda doc: [cv.vocabulary_[w] for w in doc] + [-1] * (max_sent_len - len(doc))
            )(doc) for doc in X
        ), dtype=np.int32)

        # test for the unit test, we are going down the rabbit hole
        assert X.shape == (n_documents, max_sent_len)
        time.append(datetime.now())
        print(f"loading complete!. Took {time[1] - time[0]}")
        print("starting unit test...")
        result = sliding_window_tester(X, n_vocab, n_documents, max_sent_len, n_jobs=8)
        print(result)
        time.append(datetime.now())
        print(f"sliding window took {time[2] - time[1]}")

    def test_ww_edges_time(self):
        print("loading and parsing data, this might take a few seconds...")
        time = [datetime.now()]
        train = pd.read_csv("../../data/amazon/train_40k.csv")

        X = train['Text'].tolist()
        cv = CountVectorizer(stop_words="english", min_df=5, max_df=0.9).fit(X)
        n_vocab = len(cv.vocabulary_)
        n_documents = len(X)
        X = jl.Parallel(n_jobs=8)(
            jl.delayed(
                lambda doc: [
                    x.lower() for x in RegexpTokenizer(r"\w+").tokenize(doc) if x.lower() in cv.vocabulary_
                ]
            )(doc) for doc in tqdm(X)
        )
        max_sent_len = max(map(len, X))
        X = np.array(jl.Parallel(n_jobs=8)(
            jl.delayed(
                lambda doc: [cv.vocabulary_[w] for w in doc] + [-1] * (max_sent_len - len(doc))
            )(doc) for doc in X
        ), dtype=np.int32)

        # test for the unit test, we are going down the rabbit hole
        assert X.shape == (n_documents, max_sent_len)
        time.append(datetime.now())
        print(f"loading complete!. Took {time[1] - time[0]}")
        print("starting unit test...")
        result = compute_word_word_edges(X, n_vocab, n_documents, max_sent_len, n_jobs=8)
        print(f"edge shape is {result[0].shape}")
        print(result)
        time.append(datetime.now())
        print(f"graph building took {time[2] - time[1]}")

    def test_sliding_window(self):
        # this unit test was a lot of work
        X = np.array([
            [ 0,  1,  2,  3,  4, -1, -1, -1],
            [ 5,  3,  4,  1,  2,  0,  5,  1]
        ], dtype=np.int32)
        # upper triangle of matrix, lower is is identical
        expected_cij = np.array([
             4, 3, 3, 0, 0, 2,
                6, 4, 2, 2, 1,
                   6, 2, 2, 1,
                      4, 3, 1,
                         4, 1,
                            3
        ], dtype=np.uint32)

        actual_cij = sliding_window_tester(X, 6, 2, 8, window_size=3)

        np.testing.assert_equal(expected_cij, actual_cij)

    def test_matrix(self):
        self.assertTrue(test_sym_matrix())

    def test_compute_word_word_edges(self):
        X = np.array([
            [0, 1, 2, 3, 4, -1, -1, -1],
            [5, 3, 4, 1, 2, 0, 5, 1]
        ], dtype=np.int32)
        coo, weights = compute_word_word_edges(X, 6, 2, 8, 3, verbose=2)
        print(coo)
        print(weights)


