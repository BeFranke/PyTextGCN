import string

from nltk import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
import torch as th
import numpy as np
import joblib as jl


def pmi(cv: CountVectorizer, documents, window_size, strides, n_jobs):
    vocab_size = len(cv.vocabulary_.values())

    p_i = th.zeros(vocab_size)
    p_ij = th.zeros((vocab_size, vocab_size))
    total_windows = 0
    num_documents = len(documents)
    """
    result = jl.Parallel(n_jobs=n_jobs, prefer="threads")(
        jl.delayed(pmi_document)(cv, document, window_size, strides) for i, document in enumerate(documents)
    )
    p_i, p_ij, total_windows = list(map(sum, zip(*result)))
    """
    for i, document in enumerate(documents):
        a, b, c = pmi_document(cv, document, window_size, strides)
        p_i = p_i + a
        p_ij = p_ij + b
        total_windows += c
        print(f"Document {i+1} of {len(documents)}")
    # normalization:
    p_i = p_i / total_windows
    p_ij = p_ij / total_windows

    print(p_i)
    pm_ij = th.log(th.divide(p_ij, th.outer(p_i, p_i)))  # outer product to get every ij combination
    pm_ij = th.max(pm_ij, th.FloatTensor([0]).expand_as(pm_ij))
    # set main diagonal to 1
    pm_ij[np.diag_indices(pm_ij.shape[0])] = th.ones(pm_ij.shape[0])
    return pm_ij


def pmi_document(cv, document, window_size, strides):
    # sample sentence:
    # cv = CountVectorizer(stop_words='english', min_df=1)
    # cv.fit(corpus)
    document = [x.lower() for x in RegexpTokenizer(r"\w+").tokenize(document) if x.lower() in cv.vocabulary_]
    # encode each word individually to get one-hot encoding
    encoded_sentence = cv.transform(document).todense()
    if encoded_sentence.shape[0] <= 1:
        return 0, 0, 0
    elif encoded_sentence.shape[0] < window_size:
        sliding_window = th.tensor(encoded_sentence).T[None, :, :]
    else:
        t = th.tensor(encoded_sentence)

        # sliding window over one-hot encoding of sentence
        sliding_window = t.unfold(dimension=0, size=window_size, step=strides)

    # total number of sliding windows:
    num_windows = sliding_window.shape[0]
    #  print(f"Windows: {num_windows}")
    # sum one-hot encodings over all words and windows => number of occurrences per token in vocabulary
    # = number of sliding windows that contains the token:
    # shape of sliding window: # windows x vocab x window_size
    p_i = sliding_window.sum(dim=2)
    p_i = th.min(p_i, th.tensor(1))
    p_i = p_i.sum(dim=0)

    # reduce each window to an encoding indication which tokens occur in the window
    occurrences = th.min(sliding_window.sum(dim=2), th.tensor(1))
    # sum of outer product of tokens
    # => occurence matrix (except diagonal, which is increased for each occurrence of the token)
    p_ij = th.einsum('ij,ik->jk', occurrences, occurrences)
    return p_i, p_ij, num_windows  # we need to accummulate those for each sentence
    # note: log(0) = -inf


def pmi_test():
    import pandas as pd

    path = "C:/Users/Fabi/Desktop/text_gcn/data/amazon"
    data = pd.read_csv(path + "/train.csv")
    data = data[['Text']]  # only extrat text
    data = data.dropna()

    cv = CountVectorizer(stop_words='english', min_df=5, max_features=300)
    cv.fit(data["Text"])

    result = pmi(cv, data["Text"], window_size=20, strides=1)
    print(result)


if __name__ == "__main__":
    pmi_test()
