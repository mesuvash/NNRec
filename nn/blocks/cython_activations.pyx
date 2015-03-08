import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h":
    double exp(double)

cdef double f(double z):
    return exp(z)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def cy_sigmoid(np.ndarray[double, ndim=2] z):
    cdef unsigned int NX, NY, i, j
    cdef np.ndarray[double, ndim=2] sig

    NY, NX = np.shape(z)	    
    
    sig = np.zeros((NY,NX))
    for i in xrange(NX):
        for j in xrange(NY):
            sig[j,i] = 1./(1. + exp(-z[j,i]))
    return sig

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def cy_sigmoid1d(np.ndarray[double, ndim=1] z):
    cdef unsigned int NX, i
    cdef np.ndarray[double, ndim=1] sig

    NX = len(z)       
    sig = np.zeros(NX)
    for i in xrange(NX):
            sig[i] = 1./(1. + exp(-z[i]))
    return sig

cpdef cy_tanh(z):
    return np.tanh(z)

cpdef cy_relu(z):
    z[z<0] = 0.0
    return z