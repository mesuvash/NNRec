import numpy as np
import scipy.sparse
cimport numpy as np
cimport cython
from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef cython_binarizeSparseMatrix(double[:] data, int[:] ri, int[:] rptr, int m, int n, int k, mapping):
    
    cdef:
        np.ndarray[double, ndim = 1] V
        np.ndarray[int, ndim = 1] I, J
        int ctr, i, j, l, nratings, item
        double rating

    I = np.zeros(len(data) * k,dtype=np.int32)
    J = np.zeros(len(data) * k,dtype=np.int32)
    V = np.zeros(len(data) * k)
    nratings = len(mapping)
    ctr = 0
    for i in range(m):
        for j in range(rptr[i], rptr[i + 1]):
            item = ri[j] 
            rating =  mapping[data[j]]
            for l in range(nratings):
                I[ctr] = i
                J[ctr] = item * k + l
                if rating == l:
                    V[ctr] = 1.0
                else:
                    V[ctr] = 0.1
                ctr += 1
    R = scipy.sparse.coo_matrix((V, (I, J)), shape=(m, n * k))
    R.data[R.data == 0.1] = 0.0
    return R.tocsr()

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


