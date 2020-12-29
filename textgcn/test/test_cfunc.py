from unittest import TestCase
import pandas as pd
from nltk import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from textgcn.lib import sliding_window_tester
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
        time.append(datetime.now())
        print(f"sliding window took {time[2] - time[1]}")

    def test_sliding_window(self):
        # this unit test was a lot of work
        X = np.array([
            [ 0,  1,  2,  3,  4, -1, -1, -1],
            [ 5,  3,  4,  1,  2,  0,  5,  1]
        ], dtype=np.int32)
        expected_ci = np.array([4, 6, 6, 4, 4, 3])
        expected_cij = np.array([
            [8,  3,  3, 0, 0, 2],
            [3, 12,  4, 2, 2, 1],
            [3,  4, 12, 2, 2, 1],
            [0,  2,  2, 8, 3, 1],
            [0,  2,  2, 3, 8, 1],
            [2,  1,  1, 1, 1, 6],
        ])

        actual_ci, actual_cij = sliding_window_tester(X, 6, 2, 8, window_size=3)

        np.testing.assert_equal(expected_ci, actual_ci)
        np.testing.assert_equal(expected_cij, actual_cij)
