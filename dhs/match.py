'''
Functions for matching hash sequences quickly
'''
import numpy as np
import numba

N_BITS = 16
INT_MAX = 2**(N_BITS - 1) + 1

# Construct "bits-set-table"
bits_set = np.zeros(2**N_BITS, dtype=np.uint16)

for i in xrange(2**N_BITS):
    bits_set[i] = (i & 1) + bits_set[i/2]


def ints_to_vectors(int_sequence, n_bits):
    '''
    Convert a sequence of integers into bit vector arrays

    Parameters
    ----------
    int_sequence : np.ndarray, dtype=np.int
        Sequence of integers
    n_bits : int
        Number of bits/dimensionality of bit vectors.

    Returns
    -------
    vectors : np.ndarray, dtype=np.bool
        Matrix of bit vectors, shape (len(int_sequence), n_bits)
    '''
    return np.array([[n >> i & 1 for i in range(n_bits)]
                     for n in int_sequence], np.bool)


def vectors_to_ints(vectors):
    '''
    Turn a matrix of bit vector arrays into a vector of ints

    Parameters
    ----------
    vectors : np.ndarray, dtype=np.bool
        Matrix of bit vectors, shape (n_vectors, n_bits)

    Returns
    -------
    ints : np.ndarray, dtype=np.int
        Vector of ints
    '''
    return (vectors*2**(np.arange(vectors.shape[1])*vectors)).sum(axis=1)


@numba.jit(nopython=True)
def int_dist(x, y, thresh, bits_set=bits_set):
    '''
    Compute the pairwise bit-distance matrix of two sequences of integers.

    Parameters
    ----------
    x : np.ndarray, dtype=int
        Sequence of integers
    y : np.ndarray, dtype=int
        Sequence of integers
    thresh : int
        The number of entries in the dist matrix less than or equal to this
        threshold will be returned
    bits_set : np.ndarray, dtype=int
        Table where bits_set(x) is the number of 1s in the binary
        representation of x, where x is an unsigned 16 bit int

    Returns
    -------
    distance_matrix : np.array, dtype=int
        Pairwise distance matrix of the entries in x and y
    n_below : int
        The number of entries in the distance matrix below `threshold`
    '''
    nx = x.shape[0]
    ny = y.shape[0]
    output = np.zeros((nx, ny), dtype=np.int32)
    # Keep track of the total number of distances below the threshold
    n_below = 0
    # Populate the distance matrix
    for m in xrange(nx):
        for n in xrange(ny):
            # XORing ^ x[m] and y[n] will produce a 16-bit int where the i'th
            # bit is 1 when the i'th bit of x[m] and the i'th bit of y[n] are
            # the same.  Retrieving the entry in bits_set will then count
            # the number of entries in x[m] and y[n] which are the same.
            output[m, n] = bits_set[x[m] ^ y[n]]
            # Accumulate the number of distances less than or equal to thresh
            n_below += (output[m, n] <= thresh)
    return output, n_below


@numba.jit(nopython=True)
def dtw(D, gully, pen):
    '''
    Compute the dynamic time warping distance between two sequences given a
    distance matrix.  The score is normalized by the path length.  Assumes an
    integer distance matrix.

    Parameters
    ----------
    D : np.ndarray, dtype=int
        Distances between two sequences
    gully : float
        Sequences must match up to this porportion of shorter sequence
    pen : int
        Non-diagonal move penalty

    Returns
    -------
    score : float
        DTW score of lowest cost path through the distance matrix, normalized
        by the path length.

    Notes
    -----
    `D` is modified in place.
    '''
    # Pre-allocate path length matrix
    path_length = np.ones(D.shape, dtype=np.uint16)

    # At each loop iteration, we are computing lowest cost to D[i + 1, j + 1]
    for i in xrange(D.shape[0] - 1):
        for j in xrange(D.shape[1] - 1):
            # Diagonal move (which has no penalty) is lowest
            if D[i, j] <= D[i, j + 1] + pen and D[i, j] <= D[i + 1, j] + pen:
                path_length[i + 1, j + 1] += path_length[i, j]
                D[i + 1, j + 1] += D[i, j]
            # Horizontal move (has penalty)
            elif D[i, j + 1] <= D[i + 1, j] and D[i, j + 1] + pen <= D[i, j]:
                path_length[i + 1, j + 1] += path_length[i, j + 1]
                D[i + 1, j + 1] += D[i, j + 1] + pen
            # Vertical move (has penalty)
            elif D[i + 1, j] <= D[i, j + 1] and D[i + 1, j] + pen <= D[i, j]:
                path_length[i + 1, j + 1] += path_length[i + 1, j]
                D[i + 1, j + 1] += D[i + 1, j] + pen

    # Traceback from lowest-cost point on bottom or right edge
    gully = int(gully*min(D.shape[0], D.shape[1]))
    i = np.argmin(D[gully:, -1]) + gully
    j = np.argmin(D[-1, gully:]) + gully

    if D[-1, j] > D[i, -1]:
        j = D.shape[1] - 1
    else:
        i = D.shape[0] - 1

    # Score is the final score of the best path
    score = D[i, j]/float(path_length[i, j])

    return score


def match_one_sequence(query, sequences, gully, penalty,
                       sequence_indices=None):
    '''
    Match a query sequence to one of the sequences in a list

    Parameters
    ----------
    query : np.ndarray, dtype='uint16'
        Query sequence
    sequences : list of np.ndarray, dtype='uint16'
        Sequences to find matches in, sorted by sequence length
    gully : float
        Sequences must match up to this porportion of shorter sequence
    penalty : int
        DTW Non-diagonal move penalty
    sequence_indices : iterable or None
        Iterable of the indices of entries of `sequences` which should be
        matched against.  If `None`, match against all sequences.

    Returns
    -------
    matches : list of int
        List of match indices
    scores : list of float
        Scores for each match
    n_pruned_dist : int
        Number of sequences pruned because not enough distances were below the
        current threshold
    '''
    # Pre-allocate match and score lists
    matches = []
    scores = []
    n_pruned_dist = 0
    # Keep track of the best DTW score so far; we want the starting value to be
    # sufficiently large that the DTW cost will for sure be computed for the
    # first sequence.  The DTW cost is essentially the mean distance between
    # sequence entries across the lowest-cost path.  The bitwise distance is
    # bounded by the number of bits.  So, for 256-bit integers, the largest
    # distance may be 2**8.  I doubt anyone will use larger integers than that.
    best_so_far = 2**8
    # Default: Check all sequences
    if sequence_indices is None:
        sequence_indices = xrange(len(sequences))
    for n in sequence_indices:
        # int_dist returns the distance matrix and the number of entries below
        # the supplied threshold in the distance matrix
        distance_matrix, n_below = int_dist(
            query, sequences[n], int(np.ceil(best_so_far)), bits_set)
        # If the number of entries below the ceil(best_cost_so_far) is less
        # than the min path length, don't bother computing DTW
        if n_below < min(query.shape[0], sequences[n].shape[0]):
            n_pruned_dist += 1
            score = np.inf
        else:
            # Compute DTW distance
            score = dtw(distance_matrix, gully, penalty)
            # Update the best score found so far
            if score < best_so_far:
                best_so_far = score

        # Store the score/match (even if it's not the best)
        matches.append(n)
        scores.append(score)
    # Sort the scores and matches
    sorted_idx = np.argsort(scores)
    matches = [matches[n] for n in sorted_idx]
    scores = [scores[n] for n in sorted_idx]
    return matches, scores, n_pruned_dist
