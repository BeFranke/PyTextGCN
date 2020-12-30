# cython: language_level=3

import numpy as np
cimport numpy as np

np.import_array()

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cython.parallel cimport prange
from libc.math cimport log
from libc.string cimport memset

# helper struct
cdef struct WeightedEdges:
    int* coo
    float* weights
    int n_edges

# constant declaring which is the threshold for a pmi score to be considered "not zero"
cdef float EPSILON = 1e-10

##### INTERFACE ######
cpdef tuple compute_word_word_edges(int[:, ::1] X, unsigned int n_vocab, unsigned int n_documents,
                                    unsigned int seq_len, unsigned int window_size = 20,  unsigned int n_jobs = 1,
                                    unsigned int verbose=0):
    """
    bridge function that serves a a python-to-C entry for the graph computation
    :param X: input text, cleaned and (densely!) tokenized by CountVectorizer or similar, 
                shape (n_documents, seq_len) with values in [0, n_vocab)
    :param n_vocab: vocabulary size in the input
    :param n_documents: number of documents in the input
    :param seq_len: length of sequences
    :param window_size: size of sliding context window. Higher values capture long-range dependencies 
                        at the cost of short range dependencies
    :param n_jobs: (UNUSED) number of jobs for parallel processing. Higher = more RAM and CPU usage
    :param verbose: integer in range [0, 1, 2], higher means more debug output
    :return: Tuple of (coo, weights), where:
                - coo is a memview of shape (n_edges, 2), dtype uint32
                - weights is a memview of shape (n_edges,), dtype float32
    """
    # init block
    # c_ij is symmetric, so we simulate only its upper traiangle
    cdef unsigned int* c_ij = <unsigned int*> PyMem_Malloc(sizeof(unsigned int) * SymMatSize_Diag(n_vocab))
    memset(c_ij, 0, sizeof(unsigned int) * SymMatSize_Diag(n_vocab))
    cdef unsigned int n_windows
    cdef WeightedEdges* we
    cdef np.npy_intp edge_dims[2]
    cdef np.npy_intp weight_dims[1]


    # apply sliding window
    n_windows = sliding_window(&X[0, 0], c_ij, window_size, n_vocab, n_documents,
                                                 seq_len, n_jobs)
    if verbose > 1:
        print("Sliding window was applied")

    # compute edges from the sliding window counts
    we = edges_from_counts(c_ij, n_vocab, n_windows, verbose)

    edge_dims = [we.n_edges, 2]
    weight_dims = [we.n_edges]
    if verbose > 1:
        print(f"Number of word-word-edges: {we.n_edges}")
    return np.PyArray_SimpleNewFromData(2, edge_dims, np.NPY_INT32, we.coo), \
           np.PyArray_SimpleNewFromData(1, weight_dims, np.NPY_FLOAT32, we.weights)



##### INTERNALS ######
cdef unsigned int sliding_window(const int* X, unsigned int* c_ij, unsigned int window_size,
                                 unsigned int n_vocab, unsigned int n_documents, unsigned int seq_len,
                                 unsigned int n_jobs):
    """
    applies sliding window and computes c_ij
    SIDE EFFECT: modifies c_ij
    :param X: input as in compute_graph_structure
    :param c_ij: pointer to word-pair count array to fill (size SymMatSizeDiag(n_vocab))
    :param n_vocab: vocabulary size in the input
    :param n_documents: number of documents in the input
    :param seq_len: length of sequences
    :param window_size: size of sliding context window. Higher values capture long-range dependencies 
                        at the cost of short range dependencies
    :param n_jobs: number of jobs for parallel processing. Higher = more RAM and CPU usage
    :return: number of windows
    """
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int k
    cdef unsigned int l
    cdef unsigned int n_windows = 0
    cdef unsigned int idx1
    cdef unsigned int idx2
    for i in range(n_documents):
        # this is the document loop
        for j in range(seq_len - window_size + 1):
            # this is the start of the sliding window
            if X[i * seq_len + j + window_size - 1] == -1 and j != 0:
                # sliding window contains padding values -> end of document reached
                break
            n_windows += 1
            # print(f"Document {i}, Index {j}")
            for k in range(j, j + window_size):
                # this loop iterates over the sliding window
                for l in range(k, j + window_size):
                    # this loop is for the pairwise counts
                    # this if needs to stay here, for the case that sliding window > sequence length, j == 0
                    if X[i * seq_len + k] != -1 and X[i * seq_len + l] != -1:
                        idx1 = X[i * seq_len + k]
                        idx2 = X[i * seq_len + l]
                        c_ij[SymMat_Diag_idx(idx1, idx2, n_vocab)] += 1
                    else:
                        break

    return n_windows


cdef WeightedEdges* edges_from_counts(unsigned int* c_ij, unsigned int n_vocab, unsigned int n_windows,
                                      unsigned int verbose):
    """
    computes edges and edge weights from c_ij
    SIDE EFFECTS: deallocates c_ij
    :param c_ij: symmetric count matrix, shape (SymMatSizeDiag(n_vocab,))
    :param n_vocab: number of unique words
    :param n_windows: number of sliding windows used to compute c_ij 
    :param verbose: higher = more debug output
    :return: pointer to WeightedEdges struct containing coo-edges and their weights
    """
    # init block
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int k
    # bitmap of edges, reserve just enough space to encapsulate the upper triangle of the adj-matrix without main diag
    cdef float* edge_field = <float*> PyMem_Malloc(sizeof(float) * SymMatSize_NoDiag(n_vocab))
    cdef float* p = <float*> PyMem_Malloc(sizeof(float) * n_vocab)
    cdef WeightedEdges* result = <WeightedEdges*> PyMem_Malloc(sizeof(WeightedEdges))
    cdef float p_ij
    cdef float pmi
    cdef unsigned int n_edges = 0
    cdef int* edges
    cdef float* weights
    if verbose > 1:
        print("init block complete")

    # compute p_i, c_i values are the main diagonal of c_ij
    for i in range(n_vocab):
        p[i] = <float>c_ij[SymMat_Diag_idx(i, i, n_vocab)] / <float>n_windows

    if verbose > 1:
        print("p_i computed")

    # trying to save space by computing p_ij on demand and only saving the value when there is an edge.
    for i in range(n_vocab - 1):
        # we only need to iterate the upper triangle without main diagonal
        for j in range(i + 1, n_vocab):
            p_ij = <float> c_ij[SymMat_Diag_idx(i, j, n_vocab)] / <float> n_windows
            if p_ij == 0 or p[i] == 0 or p[j] == 0:
                # log or division would cause error
                edge_field[SymMat_NoDiag_idx(i, j, n_vocab)] = 0
                continue
            pmi = log(p_ij / (p[i] * p[j]))
            if pmi > EPSILON:
                edge_field[SymMat_NoDiag_idx(i, j, n_vocab)] = pmi
                n_edges += 1
            else:
                edge_field[SymMat_NoDiag_idx(i, j, n_vocab)] = 0
    if verbose > 1:
        print("pij computed")
    # p and c_ij are no longer needed
    PyMem_Free(p)
    PyMem_Free(c_ij)
    if verbose > 1:
        print("free")
    # don't forget self-loops
    n_edges = n_edges * 2 # + n_vocab   # add n_vocab if self-loops need to be added manually

    # extract the edges and weights into a fixed size memory region
    edges = <int*> PyMem_Malloc(sizeof(int) * 2 * n_edges)
    weights = <float*> PyMem_Malloc(sizeof(int) * n_edges)
    k = 0
    for i in range(n_vocab - 1):
        for j in range(i + 1, n_vocab):
            if edge_field[SymMat_NoDiag_idx(i, j, n_vocab)] > 0:
                edges[2 * k] = i
                edges[2 * k + 1] = j
                weights[k] = edge_field[SymMat_NoDiag_idx(i, j, n_vocab)]
                k += 1
                # edges are symmetric, so we add the inverse edge too
                edges[2 * k] = j
                edges[2 * k + 1] = i
                weights[k] = edge_field[SymMat_NoDiag_idx(i, j, n_vocab)]
                k += 1
    if verbose > 1:
        print("edges converted to coo")
    # diag matrix no longer needed
    PyMem_Free(edge_field)

    # we explicitly do not add self-loops, as GCN does that already
    # add self-loops
    # i = 0
    # for k in range(k, n_edges):
    #    edges[2 * k] = i
    #    edges[2 * k + 1] = i
    #    weights[k] = 1
    #    i += 1

    result.weights = weights
    result.coo = edges
    result.n_edges = n_edges

    return result

##### HELPER FUNCTIONS #####
cdef unsigned int SymMat_Diag_idx(unsigned int row, unsigned int col, unsigned int N):
    """
    computes the index to a 1D-array when that 1D-array is used to simulate a symmetric 2D matrix
    of size (N x N) (including main diagonal) given the "simulated" indices (row, col)
    :param row: 
    :param col: 
    :param N: 
    :return: 
    """
    if row >= col:
        return col * N + row - <unsigned int>((col + 1) * col / 2)
    else:
        return row * N + col - <unsigned int>((row + 1) * row / 2)

cdef unsigned int SymMat_NoDiag_idx(unsigned int row, unsigned int col, unsigned int N):
    """
    computes the index to a 1D-array when that 1D-array is used to simulate a symmetric 2D matrix
    of size (N x N) (excluding main diagonal) given the "simulated" indices (row, col)
    :param row: 
    :param col: 
    :param N: 
    :return: 
    """
    assert row != col       # diagonal doesn't exist here
    if row > col:
        return col * N + row - <unsigned int>((col + 1) * col / 2) - col
    else:
        return row * N + col - <unsigned int>((row + 1) * row / 2) - row

cdef unsigned int SymMatSize_Diag(unsigned int N):
    """
    computes the size of an 1D array used to simulate a symmetric 2D matrix
    of size (N x N) (including main diagonal)
    :param N: 
    :return: 
    """
    return <unsigned int> ((N * (N + 1)) / 2)

cdef unsigned int SymMatSize_NoDiag(unsigned int N):
    """
    computes the size of an 1D array used to simulate a symmetric 2D matrix
    of size (N x N) (excluding main diagonal)
    :param N: 
    :return: 
    """
    return <unsigned int> ((N * (N + 1)) / 2 - N)


##### FUNCTIONS BELOW ARE ONLY FOR TESTING #####
cpdef unsigned int[::1] sliding_window_tester(int[:, ::1] X, unsigned int n_vocab, unsigned int n_documents,
                                                 unsigned int seq_len, unsigned int window_size = 20,
                                                 unsigned int n_jobs = 1):
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
    :return: c_ij containing pairwise counts 
    """
    # init block
    cdef unsigned int[::1] c_ij = np.zeros(SymMatSize_Diag(n_vocab), dtype=np.uint32)
    # apply sliding window
    cdef unsigned int n_windows = sliding_window(&X[0, 0], &c_ij[0], window_size, n_vocab, n_documents,
                                                 seq_len, n_jobs)

    return c_ij

cpdef short test_sym_matrix():
    """
    test function for the symmetric matrix functions
    :return: 
    """
    cdef float mat[10]       # simulate 4 x 4 matrix
    memset(mat, 0, sizeof(float) * 10)

    mat[SymMat_Diag_idx(1, 1, 4)] = 10
    mat[SymMat_Diag_idx(1, 2, 4)] = 20
    mat[SymMat_Diag_idx(2, 0, 4)] = 30
    mat[SymMat_Diag_idx(3, 3, 4)] = 100
    mat[SymMat_Diag_idx(2, 3, 4)] = 120

    expected = [0, 0, 30, 0, 10, 20, 0, 0, 120, 100]
    result = 1
    for i in range(6):
        result &= mat[i] == expected[i]
        print(f"expected {expected[i]}, got {mat[i]}")

    return result
