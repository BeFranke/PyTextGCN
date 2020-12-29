# cython: language_level=3

import numpy as np
cimport numpy as np

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cython.parallel cimport prange

cpdef tuple compute_graph_structure(int[:, ::1] X, unsigned int n_vocab, unsigned int n_documents,
                                    unsigned int seq_len, unsigned int window_size = 20,  unsigned int n_jobs = 1,
                                    unsigned int batch_size = 400):
    """
    bridge function that serves a a python-to-C entry for the graph computation
    :param X: input text, cleaned and (densely!) tokenized by CountVectorizer or similar, 
                shape (n_documents, seq_len) with values in [0, n_vocab)
    :param n_vocab: vocabulary size in the input
    :param n_documents: number of documents in the input
    :param seq_len: length of sequences
    :param window_size: size of sliding context window. Higher values capture long-range dependencies 
                        at the cost of short range dependencies
    :param n_jobs: number of jobs for parallel processing. Higher = more RAM and CPU usage
    :param batch_size: batch size to work on. Higher = more RAM usage
    :return: ResultObj containing edges in coo format and weights
    """
    # init block
    cdef unsigned int[::1] c_i = np.zeros(n_vocab, dtype=np.uint32)
    cdef unsigned int[:, ::1] c_ij = np.zeros((n_vocab, n_vocab), dtype=np.uint32)
    cdef float[::1] p_i = np.zeros(n_vocab, dtype=np.float32)
    cdef float[:, ::1] p_ij = np.zeros((n_vocab, n_vocab), dtype=np.float32)

    # apply sliding window
    cdef unsigned int n_windows = sliding_window(&X[0, 0], &c_i[0], &c_ij[0, 0], window_size, n_vocab, n_documents,
                                                 seq_len, n_jobs)


    return np.asarray(p_i, dtype=np.float32), np.asarray(p_ij, dtype=np.float32)


cdef unsigned int sliding_window(const int* X, unsigned int* c_i, unsigned int* c_ij, unsigned int window_size,
                                 unsigned int n_vocab, unsigned int n_documents, unsigned int seq_len,
                                 unsigned int n_jobs):
    """
    :param X: input, not modified, as in compute_graph_structure
    :param c_i: pointer to raw word count array to fill (size n_vocab)
    :param c_ij: pointer to word-pair count array to fill (size n_vocab^2)
    :param n_vocab: vocabulary size in the input
    :param n_documents: number of documents in the input
    :param seq_len: length of sequences
    :param window_size: size of sliding context window. Higher values capture long-range dependencies 
                        at the cost of short range dependencies
    :param n_jobs: number of jobs for parallel processing. Higher = more RAM and CPU usage
    :return: number of windows
    SIDE EFFECT: modifies c_i and c_ij
    """
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int k
    cdef unsigned int l
    cdef unsigned int idx1
    cdef unsigned int idx2
    cdef unsigned int n_windows = 0
    for i in range(n_documents):
        # this is the document loop
        # TODO check if there is any sensible way to parallelize this
        # TODO if not its not tragical, on Amazon 40k this takes ~10s
        for j in range(seq_len - window_size + 1):
            # this is the start of the sliding window
            if X[i * seq_len + j + window_size - 1] == -1 and j != 0:
                # sliding window contains padding values -> end of document reached
                break
            n_windows += 1
            for k in range(j, j + window_size):
                # this loop iterates over the sliding window
                # update c_i
                c_i[X[i * seq_len + k]] += 1
                for l in range(k, j + window_size):
                    # this loop is for the pairwise counts
                    # this if needs to stay here, for the case that sliding window > sequence length, j == 0
                    if X[i * seq_len + k] != -1:
                        idx1 = X[i * seq_len + k] * n_vocab + X[i * seq_len + l]
                        idx2 = X[i * seq_len + k] + n_vocab * X[i * seq_len + l]
                        # we effectively compute the same score twice, but this shouldn't be an issue and make the code
                        # easier later
                        c_ij[idx1] += 1
                        c_ij[idx2] += 1
                    else:
                        break

    return n_windows


cdef void edges_from_counts(float* p_i, float* p_ij, const unsigned int* c_i, const unsigned int* c_ij,
                            const unsigned int n_vocab):
    pass


cpdef tuple sliding_window_tester(int[:, ::1] X, unsigned int n_vocab, unsigned int n_documents,
                                       unsigned int seq_len, unsigned int window_size = 20,  unsigned int n_jobs = 1,
                                       unsigned int batch_size = 400):
    """
    THIS FUNCTION SOLELY EXISTS TO EXPOSE THE RESULTS OF sliding_window, IT IS ONLY FOR TESTING
    :param X: input text, cleaned and (densely!) tokenized by CountVectorizer or similar, 
                shape (n_documents, seq_len) with values in [0, n_vocab)
    :param n_vocab: vocabulary size in the input
    :param n_documents: number of documents in the input
    :param seq_len: length of sequences
    :param window_size: size of sliding context window. Higher values capture long-range dependencies 
                        at the cost of short range dependencies
    :param n_jobs: number of jobs for parallel processing. Higher = more RAM and CPU usage
    :param batch_size: batch size to work on. Higher = more RAM usage
    :return: ResultObj containing edges in coo format and weights
    """
    # init block
    cdef unsigned int[::1] c_i = np.zeros(n_vocab, dtype=np.uint32)
    cdef unsigned int[:, ::1] c_ij = np.zeros((n_vocab, n_vocab), dtype=np.uint32)
    # apply sliding window
    cdef unsigned int n_windows = sliding_window(&X[0, 0], &c_i[0], &c_ij[0, 0], window_size, n_vocab, n_documents,
                                                 seq_len, n_jobs)
    # abuse the result object to return intermediate results
    # res.edges_coo = np.asarray(c_ij, dtype=np.float32)
    # res.edge_weights = np.asarray(c_i, dtype=np.float32)
    return np.asarray(c_i, dtype=np.int32), np.asarray(c_ij, dtype=np.int32)