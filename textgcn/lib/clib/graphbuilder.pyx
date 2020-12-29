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
    unsigned int* coo
    float* weights
    int n_edges


##### INTERFACE ######
cpdef tuple compute_word_word_edges(int[:, ::1] X, unsigned int n_vocab, unsigned int n_documents,
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
    cdef int edge_dims[2]
    cdef int weight_dims[1]


    # apply sliding window
    n_windows = sliding_window(&X[0, 0], c_ij, window_size, n_vocab, n_documents,
                                                 seq_len, n_jobs)

    # compute edges from the sliding window counts
    we = edges_from_counts(c_ij, n_vocab, n_windows)

    edge_dims = [we.n_edges, 2]
    weight_dims = [we.n_edges]

    return np.PyArray_SimpleNewFromData(2, edge_dims, np.float32, we.coo), \
           np.PyArray_SimpleNewFromData(1, weight_dims, np.int32, we.weights)



##### INTERNALS ######
cdef unsigned int sliding_window(const int* X, unsigned int* c_ij, unsigned int window_size,
                                 unsigned int n_vocab, unsigned int n_documents, unsigned int seq_len,
                                 unsigned int n_jobs):
    """
    applies sliding window and computes c_ij
    SIDE EFFECT: modifies c_ij
    :param X: input, not modified, as in compute_graph_structure
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
            # print(f"Document {i}, Index {j}")
            for k in range(j, j + window_size):
                # this loop iterates over the sliding window
                for l in range(k, j + window_size):
                    # this loop is for the pairwise counts
                    # this if needs to stay here, for the case that sliding window > sequence length, j == 0
                    if X[i * seq_len + k] != -1 and X[i * seq_len + l] != -1:
                        uint_SymMat_Inc_Diag(i, j, n_vocab, c_ij)
                    else:
                        break

    return n_windows


cdef WeightedEdges* edges_from_counts(unsigned int* c_ij, unsigned int n_vocab, unsigned int n_windows):
    """
    computes edges and edge weights from c_i and c_ij
    SIDE EFFECTS: deallocates c_ij
    :param c_ij: count matrix, shape (n_vocab, n_vocab)
    :param n_vocab: number of unique words
    :param n_windows: number of sliding windows used to compute c_ij 
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
    cdef unsigned int* edges
    cdef float* weights

    # compute p_i, c_i values are the main diagonal of c_ij
    for i in range(n_vocab):
        p[i] = <float>c_ij[i * n_vocab + i] / <float>n_windows

    # trying to save space by computing p_ij on demand and only saving the value when there is an edge.
    for i in range(n_vocab - 1):
        # we only need to iterate the upper triangle without main diagonal
        for j in range(i + 1, n_vocab):
            p_ij = <float>uint_SymMat_Get_Diag(i, j, n_vocab, c_ij) / <float>n_windows
            if p_ij == 0 or p[i] == 0 or p[j] == 0:
                # log or division would cause error
                float_SymMat_Set_NoDiag(i, j, n_vocab, edge_field, 0)
                continue
            pmi = log(p_ij / (p[i] * p[j]))
            if pmi > 0:
                # edge_field[edge_num] = pmi
                float_SymMat_Set_NoDiag(i, j, n_vocab, edge_field, pmi)
                n_edges += 1

    # p and c_ij are no longer needed
    PyMem_Free(p)
    PyMem_Free(c_ij)

    # don't forget self.loops and symmetric edges
    n_edges = n_edges * 2 + n_vocab

    # extract the edges and weights into a fixed size memory region
    edges = <unsigned int*> PyMem_Malloc(sizeof(int) * 2 * n_edges)
    weights = <float*> PyMem_Malloc(sizeof(int) * n_edges)
    k = 0
    for i in range(n_vocab - 1):
        for j in range(i + 1, n_vocab):
            if float_SymMat_Get_NoDiag(i, j, n_vocab, edge_field) > 0:
                edges[2 * k] = i
                edges[2 * k + 1] = j
                weights[k] = float_SymMat_Get_NoDiag(i, j, n_vocab, edge_field)
                k += 1
                edges[2 * k] = j
                edges[2 * k + 1] = i
                weights[k] = float_SymMat_Get_NoDiag(i, j, n_vocab, edge_field)

    # diag matrix no longer needed
    PyMem_Free(edge_field)

    # add self-loops
    for k in range(k, n_edges):
        edges[2 * k] = i
        edges[2 * k + 1] = i
        weights[k] = 1

    result.weights = weights
    result.coo = edges
    result.n_edges = n_edges

    return result

##### HELPER FUNCTIONS #####
cdef float float_SymMat_Get_Diag(unsigned int row, unsigned int col, unsigned int N, float* mat):
    # taken from
    # https://www.codeguru.com/cpp/cpp/algorithms/general/article.php/c11211/TIP-Half-Size-Triangular-Matrix.htm
    if row > col:
        return mat[<int>(row * N - (row - 1) * row / 2 + col - row)]
    else:
        return mat[<int>(row * N - (row - 1) * row / 2 + col - row)]

cdef void float_SymMat_Set_Diag(unsigned int row, unsigned int col, unsigned int N, float* mat, float e):
    # taken from
    # https://www.codeguru.com/cpp/cpp/algorithms/general/article.php/c11211/TIP-Half-Size-Triangular-Matrix.htm
    if row > col:
        mat[<int>(row * N - (row - 1) * row / 2 + col - row)] = e
    else:
        mat[<int>(row * N - (row - 1) * row / 2 + col - row)] = e

cdef void float_SymMat_Inc_Diag(unsigned int row, unsigned int col, unsigned int N, float* mat):
    # taken from
    # https://www.codeguru.com/cpp/cpp/algorithms/general/article.php/c11211/TIP-Half-Size-Triangular-Matrix.htm
    if row > col:
        mat[<int>(row * N - (row - 1) * row / 2 + col - row)] += 1
    else:
        mat[<int>(row * N - (row - 1) * row / 2 + col - row)] += 1

cdef float float_SymMat_Get_NoDiag(unsigned int row, unsigned int col, unsigned int N, float* mat):
    # taken from
    # https://www.codeguru.com/cpp/cpp/algorithms/general/article.php/c11211/TIP-Half-Size-Triangular-Matrix.htm
    assert row != col
    if row >= col:
        return mat[<int>(row * (N - 1) - (row - 1) * row / 2 + col - row - 1)]
    else:
        return mat[<int>(row * (N - 1) - (row - 1) * row / 2 + col - row - 1)]

cdef void float_SymMat_Set_NoDiag(unsigned int row, unsigned int col, unsigned int N, float* mat, float e):
    # taken from
    # https://www.codeguru.com/cpp/cpp/algorithms/general/article.php/c11211/TIP-Half-Size-Triangular-Matrix.htm
    assert row != col
    if row >= col:
        mat[<int>(row * (N - 1) - (row - 1) * row / 2 + col - row - 1)] = e
    else:
        mat[<int>(row * (N - 1) - (row - 1) * row / 2 + col - row - 1)] = e

cdef void float_SymMat_Inc_NoDiag(unsigned int row, unsigned int col, unsigned int N, float* mat):
    # taken from
    # https://www.codeguru.com/cpp/cpp/algorithms/general/article.php/c11211/TIP-Half-Size-Triangular-Matrix.htm
    assert row != col
    if row >= col:
        mat[<int>(row * (N - 1) - (row - 1) * row / 2 + col - row - 1)] += 1
    else:
        mat[<int>(row * (N - 1) - (row - 1) * row / 2 + col - row - 1)] += 1

cdef unsigned int uint_SymMat_Get_Diag(unsigned int row, unsigned int col, unsigned int N, unsigned int* mat):
    # taken from
    # https://www.codeguru.com/cpp/cpp/algorithms/general/article.php/c11211/TIP-Half-Size-Triangular-Matrix.htm
    if row > col:
        return mat[<int>(row * N - (row - 1) * row / 2 + col - row)]
    else:
        return mat[<int>(row * N - (row - 1) * row / 2 + col - row)]

cdef void uint_SymMat_Set_Diag(unsigned int row, unsigned int col, unsigned int N, unsigned int* mat, unsigned int e):
    # taken from
    # https://www.codeguru.com/cpp/cpp/algorithms/general/article.php/c11211/TIP-Half-Size-Triangular-Matrix.htm
    if row > col:
        mat[<int>(row * N - (row - 1) * row / 2 + col - row)] = e
    else:
        mat[<int>(row * N - (row - 1) * row / 2 + col - row)] = e

cdef void uint_SymMat_Inc_Diag(unsigned int row, unsigned int col, unsigned int N, unsigned int* mat):
    # taken from
    # https://www.codeguru.com/cpp/cpp/algorithms/general/article.php/c11211/TIP-Half-Size-Triangular-Matrix.htm
    if row > col:
        mat[<int>(row * N - (row - 1) * row / 2 + col - row)] += 1
    else:
        mat[<int>(row * N - (row - 1) * row / 2 + col - row)] += 1

cdef unsigned int uint_SymMat_Get_NoDiag(unsigned int row, unsigned int col, unsigned int N, unsigned int* mat):
    # taken from
    # https://www.codeguru.com/cpp/cpp/algorithms/general/article.php/c11211/TIP-Half-Size-Triangular-Matrix.htm
    assert row != col
    if row >= col:
        return mat[<int>(row * (N - 1) - (row - 1) * row / 2 + col - row - 1)]
    else:
        return mat[<int>(row * (N - 1) - (row - 1) * row / 2 + col - row - 1)]

cdef void uint_SymMat_Set_NoDiag(unsigned int row, unsigned int col, unsigned int N, unsigned int* mat, unsigned int e):
    # taken from
    # https://www.codeguru.com/cpp/cpp/algorithms/general/article.php/c11211/TIP-Half-Size-Triangular-Matrix.htm
    assert row != col
    if row >= col:
        mat[<int>(row * (N - 1) - (row - 1) * row / 2 + col - row - 1)] = e
    else:
        mat[<int>(row * (N - 1) - (row - 1) * row / 2 + col - row - 1)] = e

cdef void uint_SymMat_Inc_NoDiag(unsigned int row, unsigned int col, unsigned int N, unsigned int* mat):
    # taken from
    # https://www.codeguru.com/cpp/cpp/algorithms/general/article.php/c11211/TIP-Half-Size-Triangular-Matrix.htm
    assert row != col
    if row >= col:
        mat[<int>(row * (N - 1) - (row - 1) * row / 2 + col - row - 1)] += 1
    else:
        mat[<int>(row * (N - 1) - (row - 1) * row / 2 + col - row - 1)] += 1

cdef unsigned int SymMatSize_Diag(unsigned int N):
    return <unsigned int> ((N * (N - 1)) / 2)

cdef unsigned int SymMatSize_NoDiag(unsigned int N):
    return <unsigned int> ((N * (N - 1)) / 2 - N)


##### FUNCTIONS BELOW ARE ONLY FOR TESTING #####
cpdef unsigned int[:, ::1] sliding_window_tester(int[:, ::1] X, unsigned int n_vocab, unsigned int n_documents,
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
    cdef unsigned int[:, ::1] c_ij = np.zeros((n_vocab, n_vocab), dtype=np.uint32)
    # apply sliding window
    cdef unsigned int n_windows = sliding_window(&X[0, 0], &c_ij[0, 0], window_size, n_vocab, n_documents,
                                                 seq_len, n_jobs)
    # abuse the result object to return intermediate results
    # res.edges_coo = np.asarray(c_ij, dtype=np.float32)
    # res.edge_weights = np.asarray(c_i, dtype=np.float32)
    return c_ij
