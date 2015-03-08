import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef np.ndarray[double, ndim = 2] multiplyWithSelectedIndices(double[:, :] x,
                                                               int[:] indices,
                                                               double[:, :] w):
    cdef:
        np.ndarray[double, ndim = 2] result
        int i, n, m, j

    m = x.shape[1]
    n = w.shape[1]

    result = np.zeros((1, n))

    for i in range(n):
        for j in range(m):
            result[0, i] = result[0, i] + w[indices[j], i] * x[0, j]
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef np.ndarray[double, ndim = 2] multiplyWithSelectedIndicesTrans(double[:, :] x,
                                                                    int[:] indices,
                                                                    double[:, :] w):
    cdef:
        np.ndarray[double, ndim = 2] result
        int i, n, m, j, index

    m = x.shape[1]
    n = len(indices)

    result = np.zeros((1, n))

    for i in range(n):
        index = indices[i]
        for j in range(m):
            result[0, i] = result[0, i] + w[j, index] * x[0, j]
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef np.ndarray[double, ndim = 1] multiplyOuterSparseLayer(double[:, :] hiddenActivation,
                                                            double[:, :] W2,
                                                            double[:, :] vis_bias,
                                                            double[:] data,
                                                            int[:] indices,
                                                            int[:] indptr,
                                                            int num_threads):
    cdef:
        np.ndarray[double, ndim = 1] result
        int i, j, k, l, m, n, start, end
        double _buffer

    m = hiddenActivation.shape[0]
    n = hiddenActivation.shape[1]
    result = np.zeros(len(data))
    for i in prange(m,  nogil=True, num_threads=num_threads):
        start = indptr[i]
        end = indptr[i + 1]
        for j in range(indptr[i + 1] - indptr[i]):
            l = indices[start]
            _buffer = 0.0
            for k in range(n):
                _buffer = _buffer + hiddenActivation[i, k] * W2[k, l]
            result[start] = _buffer + vis_bias[0, l]
            start = start + 1
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef np.ndarray[double, ndim = 2] addWithSelectedIndices(double[:, :] x,
                                                          int[:] indices,
                                                          double[:, :] y):
    cdef:
        np.ndarray[double, ndim = 2] result
        int i,  m

    m = x.shape[1]

    result = np.zeros((1, m))

    for i in range(m):
        result[0, i] = x[0, i] + y[0, indices[i]]
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef addMatricesWithSelectedIndices(double[:, :] w1,
                                     int[:] indices, int indexAxis,
                                     double[:, :] w2):
    cdef:
        int i, j, m, n, index

    m = w2.shape[0]
    n = w2.shape[1]

    if (indexAxis == 1):
        for j in range(n):
            index = indices[j]
            for i in range(m):
                w1[i, index] = w1[i, index] + w2[i, j]
    else:
        for i in range(m):
            index = indices[i]
            for j in range(n):
                w1[index, j] = w1[index, j] + w2[i, j]



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef addMatricesWithSelectedIndicesSGD(double[:, :] w1,
                                        int[:] indices, int indexAxis,
                                        double[:, :] w2,
                                        double learn_rate,
                                        double decay):

    cdef:
        int i, j, m, n, index

    m = w2.shape[0]
    n = w2.shape[1]

    if (indexAxis == 1):
        for j in range(n):
            index = indices[j]
            for i in range(m):
                w1[i, index] = w1[i, index] - learn_rate * \
                    (w2[i, j] + decay * w1[i, index])
    else:
        for i in range(m):
            index = indices[i]
            for j in range(n):
                w1[index, j] = w1[index, j] - learn_rate * \
                    (w2[i, j] + decay * w1[index, j])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef dropout(double[:, :] mat, double p):
    cdef:
        int i, j, m, n, numdrop
        np.ndarray[long, ndim = 1] indices
    m = mat.shape[0]
    n = mat.shape[1]
    numdrop = int(n * p)
    for i in range(m):
        indices = np.random.choice(n, numdrop, replace=False)
        for j in indices:
            mat[i, j] = 0.0
