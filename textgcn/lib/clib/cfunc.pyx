import numpy as np
cimport numpy as np

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cython.parallel cimport prange

cpdef struct ResultObj:
    double[:] edge_weights
    double[:, :] edges_coo

cpdef ResultObj* compute_graph_structure(unsigned int[:, :] X, unsigned int n_vocab, unsigned int n_documents,
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
    cdef ResultObj* res = <ResultObj*>PyMem_Malloc(sizeof(ResultObj))
    cdef unsigned int[:] c_i = np.zeros(n_vocab)
    cdef unsigned int[:, :] c_ij = np.zeros((n_vocab, n_vocab))

    # apply sliding window
    cdef unsigned int n_windows = sliding_window(&X, &c_i, &c_ij, window_size, n_vocab, n_documents, seq_len, n_jobs)


cdef unsigned int sliding_window(const unsigned int* X, unsigned int* c_i, unsigned int* c_ij, unsigned int window_size,
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
        # this is the document loop, which should be parallelized
        # TODO HOW? somehow parallelism needs to be ordered so that no synchronization issues occur
        # TODO alternative: one count-matrix-pair per thread and then sum
        for j in range(seq_len - window_size + 1):
            # this is the start of the sliding window
            n_windows += 1
            for k in range(j, j + window_size):
                # this loop iterates over the sliding window
                # update c_i
                c_i[X[i, k]] += 1
                for l in range(k, j + window_size):
                    # this loop is for the pairwise counts
                    idx1 = X[i, k] * n_vocab + X[i, l]
                    idx2 = X[i, k] + n_vocab * X[i, l]
                    # we effectively compute the same score twice, but this shouldn't be an issue and make the code
                    # easier later
                    c_ij[idx1] += 1
                    c_ij[idx2] += 1

    return n_windows

cdef void edges_from_counts(ResultObj* res, const unsigned int* c_i, const unsigned int* c_ij,
                            const unsigned int n_vocab)
