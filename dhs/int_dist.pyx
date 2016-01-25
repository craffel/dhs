"""
Functions for computing the pairwise bitwise distance matrix of two sequences
of integers.
For fastest results, compile with gcc flag -mpopcnt
"""
import numpy as np
cimport numpy as np
cimport cython
from libc.stdint cimport uint32_t, uint64_t, int64_t

cdef extern int __builtin_popcount(unsigned int) nogil
cdef extern int __builtin_popcountll(unsigned long long) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _int_dist_32(uint32_t[:] x, uint32_t[:] y, int64_t thresh,
                      uint32_t[:, :] output) nogil:
    """
    Computes the pairwise bitwise distance matrix between the 32-bit integer
    vectors `x` and `y` by passing their exclusive-or to the builtin popcount
    operation, and stores the result in `output`.  Also returns the total
    number of elements in the resulting distance matrix below `thresh`.

    Parameters
    ----------
    x : np.ndarray, dtype=uint32, shape=(M,)
        Sequence of integers
    y : np.ndarray, dtype=uint32, shape=(N,)
        Sequence of integers
    thresh : int
        The number of entries in the dist matrix less than or equal to this
        threshold will be returned
    output : np.ndarray, dtype=int, shape=(M, N)
        Numpy array where the distance matrix will be written.

    Returns
    -------
    n_below : int
        The number of entries in the distance matrix below `threshold`
    """
    cdef int i
    cdef int j
    # Keep track of the total number of distances below the threshold
    cdef int n_below = 0
    # Populate the distance matrix
    for i in xrange(x.shape[0]):
        for j in xrange(y.shape[0]):
            # XORing ^ x[m] and y[n] will produce a 32-bit int where the i'th
            # bit is 1 when the i'th bit of x[m] and the i'th bit of y[n] are
            # the same.  Calling the builtin popcount operation will then count
            # the number of entries in x[m] and y[n] which are the same.
            output[i, j] = __builtin_popcount(x[i] ^ y[j])
            # Accumulate the number of distances less than or equal to thresh
            n_below += (output[i, j] <= thresh)
    return n_below


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _int_dist_64(uint64_t[:] x, uint64_t[:] y, int64_t thresh,
                      uint32_t[:, :] output) nogil:
    """
    Computes the pairwise bitwise distance matrix between the 64-bit integer
    vectors `x` and `y` by passing their exclusive-or to the builtin popcount
    operation, and stores the result in `output`.  Also returns the total
    number of elements in the resulting distance matrix below `thresh`.

    Parameters
    ----------
    x : np.ndarray, dtype=uint64, shape=(M,)
        Sequence of integers
    y : np.ndarray, dtype=uint64, shape=(N,)
        Sequence of integers
    thresh : int
        The number of entries in the dist matrix less than or equal to this
        threshold will be returned
    output : np.ndarray, dtype=int, shape=(M, N)
        Numpy array where the distance matrix will be written.

    Returns
    -------
    n_below : int
        The number of entries in the distance matrix below `threshold`
    """
    cdef int i
    cdef int j
    # Keep track of the total number of distances below the threshold
    cdef int n_below = 0
    # Populate the distance matrix
    for i in xrange(x.shape[0]):
        for j in xrange(y.shape[0]):
            # XORing ^ x[m] and y[n] will produce a 64-bit int where the i'th
            # bit is 1 when the i'th bit of x[m] and the i'th bit of y[n] are
            # the same.  Calling the builtin popcount operation will then count
            # the number of entries in x[m] and y[n] which are the same.
            output[i, j] = __builtin_popcountll(x[i] ^ y[j])
            # Accumulate the number of distances less than or equal to thresh
            n_below += (output[i, j] <= thresh)
    return n_below


def int_dist(x, y, thresh):
    """
    Computes the pairwise bitwise distance matrix between the integer
    vectors `x` and `y` by passing their exclusive-or to the builtin popcount
    operation, and returns the result.  Also returns the total number of
    elements in the resulting distance matrix below `thresh`.

    Parameters
    ----------
    x : np.ndarray, shape=(M,)
        Sequence of integers
    y : np.ndarray, shape=(N,)
        Sequence of integers.  Must have the same (integer) dtype as `x`.
    thresh : int
        The number of entries in the dist matrix less than or equal to this
        threshold will be returned

    Returns
    -------
    distance_matrix : np.ndarray, dtype=int, shape=(M, N)
        Pairwise bitwisedistance matrix of the entries in x and y
    n_below : int
        The number of entries in the distance matrix below `threshold`
    """
    if not x.dtype == y.dtype:
        raise ValueError("x and y must have the same dtype, but x.dtype={} and"
                         " y.dtype={}".format(x.dtype, y.dtype))
    # Pre-allocate the distance matrix
    distance_matrix = np.empty((x.shape[0], y.shape[0]), dtype=np.uint32)
    # Choose the correct __builtin_popcount depending on dtype
    if x.dtype == np.uint32:
        n_below = _int_dist_32(x, y, thresh, distance_matrix)
    elif x.dtype == np.uint64:
        n_below = _int_dist_64(x, y, thresh, distance_matrix)
    elif not np.issubdtype(x.dtype, np.integer):
        raise ValueError("dtype {} not supported.".format(x.dtype))
    else:
        if x.nbytes / x.size < 4:
            # Allow integer dtypes to be used with fewer than 32 bits by using
            # astype.  We can't use view here because if the array doesn't have
            # a total size which is divisible by 32 bits, it will raise an
            # error.
            x_32 = x.astype(np.uint32, copy=False)
            y_32 = y.astype(np.uint32, copy=False)
            n_below = _int_dist_32(x_32, y_32, thresh, distance_matrix)
        # Use np.view for anything with 32 or 64 bits
        elif x.nbytes / x.size == 4:
            n_below = _int_dist_32(
                x.view(np.uint32), y.view(np.uint32), thresh, distance_matrix)
        elif x.nbytes / x.size == 8:
            n_below = _int_dist_64(
                x.view(np.uint64), y.view(np.uint64), thresh, distance_matrix)
        # We don't support any datatypes with > 64 bits per element
        else:
            raise ValueError("dtype {} not supported.".format(x.dtype))
    return distance_matrix, n_below
