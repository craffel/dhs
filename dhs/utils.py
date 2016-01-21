'''
Utilities for cross-modality hashing experiments, including data sampling and
statistics
'''
import numpy as np


def sample_sequences(X, Y, sample_size):
    ''' Given lists of sequences, crop out sequences of length sample_size
    from each sequence with a random offset

    Parameters
    ----------
    X, Y : list of np.ndarray
        List of X/Y sequence matrices, each with shape (n_channels,
        n_time_steps, n_features)
    sample_size : int
        The size of the cropped samples from the sequences

    Returns
    -------
    X_sampled, Y_sampled : np.ndarray
        X/Y sampled sequences, shape (n_samples, n_channels, n_time_steps,
        n_features)
    '''
    X_sampled = []
    Y_sampled = []
    for sequence_X, sequence_Y in zip(X, Y):
        # Ignore sequences which are too short
        if sequence_X.shape[1] < sample_size:
            continue
        # Compute a random offset to start cropping from
        offset = np.random.randint(0, sequence_X.shape[1] % sample_size + 1)
        # Extract samples of this sequence at offset, offset + sample_size,
        # offset + 2*sample_size ... until the end of the sequence
        X_sampled += [sequence_X[:, o:o + sample_size] for o in
                      np.arange(offset, sequence_X.shape[1] - sample_size + 1,
                                sample_size)]
        Y_sampled += [sequence_Y[:, o:o + sample_size] for o in
                      np.arange(offset, sequence_Y.shape[1] - sample_size + 1,
                                sample_size)]
    # Combine into new output array
    return np.array(X_sampled), np.array(Y_sampled)


def random_derangement(n):
    '''
    Permute the numbers up to n such that no number remains in the same place

    Parameters
    ----------
    n : int
        Upper bound of numbers to permute from

    Returns
    -------
    v : np.ndarray, dtype=int
        Derangement indices

    Note
    ----
        From
        http://stackoverflow.com/questions/26554211/numpy-shuffle-with-constraint
    '''
    while True:
        v = np.arange(n)
        for j in np.arange(n - 1, -1, -1):
            p = np.random.randint(0, j+1)
            if v[p] == j:
                break
            else:
                v[j], v[p] = v[p], v[j]
        else:
            if v[0] != 0:
                return v


def get_next_batch(X, Y, batch_size, sample_size, n_iter):
    ''' Randomly generates positive and negative example minibatches

    Parameters
    ----------
    X, Y : list of np.ndarray
        List of data matrices in each modality
    batch_size : int
        Size of each minibatch to grab
    sample_size : int
        Size of each sampled sequence
    n_iter : int
        Total number of iterations to run

    Returns
    -------
    X_p : np.ndarray
        Positive example minibatch in X modality
    Y_p : np.ndarray
        Positive example minibatch in Y modality
    X_n : np.ndarray
        Negative example minibatch in X modality
    Y_n : np.ndarray
        Negative example minibatch in Y modality
    '''
    # These are dummy values which will force the sequences to be sampled
    current_batch = 1
    # We'll only know the number of batches after we sample sequences
    n_batches = 0
    for n in xrange(n_iter):
        if current_batch >= n_batches:
            X_sampled, Y_sampled = sample_sequences(X, Y, sample_size)
            N = X_sampled.shape[0]
            n_batches = int(np.floor(N/float(batch_size)))
            # Shuffle X_p and Y_p the same
            positive_shuffle = np.random.permutation(N)
            X_p = np.array(X_sampled[positive_shuffle])
            Y_p = np.array(Y_sampled[positive_shuffle])
            # Shuffle X_n and Y_n differently (derangement ensures nothing
            # stays in the same place)
            negative_shuffle = np.random.permutation(N)
            X_n = np.array(X_sampled[negative_shuffle])
            Y_n = np.array(Y_sampled[negative_shuffle][random_derangement(N)])
            current_batch = 0
        batch = slice(current_batch*batch_size, (current_batch + 1)*batch_size)
        yield X_p[batch], Y_p[batch], X_n[batch], Y_n[batch]
        current_batch += 1


def hash_entropy(X):
    ''' Get the entropy of the histogram of hashes.
    We want this to be close to n_bits.

    Parameters
    ----------
    X : np.ndarray, shape=(n_examples, n_bits)
        Boolean data matrix, each column is the hash of an example

    Returns
    -------
    hash_entropy : float
        Entropy of the hash distribution
    '''
    # Convert bit vectors to ints
    bit_values = np.sum(2**np.arange(X.shape[1])*X, axis=1)
    # Count the number of occurences of each int
    counts = np.bincount(bit_values)
    # Normalize to form a probability distribution
    counts = counts/float(counts.sum())
    # Compute entropy
    return -np.sum(counts*np.log2(counts + 1e-100))


def statistics(X, Y):
    ''' Computes the number of correctly encoded codeworks and the number of
    bit errors made.  Assumes that rows of X should be hashed the same as rows
    of Y

    Parameters
    ----------
    X : np.ndarray, shape=(n_examples, n_features)
        Data matrix of X modality
    Y : np.ndarray, shape=(n_examples, n_features)
        Codeword matrix of Y modality

    Returns
    -------
    distance_distribution : int
        Emprical distribution of the codeword distances
    mean_distance : float
        Mean of distances between corresponding codewords
    std_distance : float
        Std of distances between corresponding codewords
    '''
    points_equal = (X == Y)
    distances = np.logical_not(points_equal).sum(axis=1)
    counts = np.bincount(distances, minlength=X.shape[1] + 1)
    return counts/float(X.shape[0]), np.mean(distances), np.std(distances)
