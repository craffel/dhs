'''
Utilities for cross-modality hashing experiments, including data loading and
statistics
'''
import numpy as np
import pickle
import lasagne


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


def build_network(input_shape, input_mean, input_std, num_filters, filter_size,
                  ds, hidden_layer_sizes, dropout, n_bits):
    '''
    Construct a list of layers of a network given the network's structure.

    Parameters
    ----------
    input_shape : tuple
        In what shape will data be supplied to the network?
    input_mean : np.ndarray
        Training set mean, to standardize inputs with.
    input_std : np.ndarray
        Training set standard deviation, to standardize inputs with.
    num_filters : list
        Number of filters in each convolutional layer
    filter_size : list
        Size of each filter in each convolutional layer
    ds : list
        Size of max-pooling window in each pooling layer
    hidden_layer_sizes : list
        Size of each hidden layer
    dropout : bool
        Should dropout be applied between fully-connected layers?
    n_bits : int
        Output dimensionality

    Returns
    -------
    layers : list
        List of layer instances for this network.
    '''
    layers = [lasagne.layers.InputLayer(shape=input_shape)]
    layers.append(lasagne.layers.standardize(
        layers[-1], input_mean, input_std, shared_axes=(0, 2)))
    # Add each convolutional and pooling layer recursively
    for n in xrange(len(num_filters)):
        # In order to get the same shape out from each convolutional layer, we
        # need to pad the input because Conv2DLayer uses 'valid' convolution by
        # default and we can't rely on Conv2DLayer's mode='same' because it
        # doesn't allow even filter sizes.
        layers.append(lasagne.layers.PadLayer(
            layers[-1], width=((int(np.ceil((filter_size[n][0] - 1) / 2.)),
                                int(np.floor((filter_size[n][0] - 1) / 2.))),
                               (int(np.ceil((filter_size[n][1] - 1) / 2.)),
                                int(np.floor((filter_size[n][1] - 1) / 2.))))))
        # We will initialize weights to \sqrt{2/n_l}
        n_l = num_filters[n]*np.prod(filter_size[n])
        layers.append(lasagne.layers.Conv2DLayer(
            layers[-1], stride=(1, 1), num_filters=num_filters[n],
            filter_size=filter_size[n],
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(np.sqrt(2./n_l))))
        layers.append(lasagne.layers.MaxPool2DLayer(
            layers[-1], ds[n], ignore_border=False))
    # A dense layer will treat any dimensions after the first as feature
    # dimensions, but the third dimension is really a timestep dimension.
    # We can only squash adjacent dimensions with a ReshapeLayer, so we
    # need to place the time stpe dimension after the batch dimension
    layers.append(lasagne.layers.DimshuffleLayer(
        layers[-1], (0, 2, 1, 3)))
    conv_output_shape = layers[-1].output_shape
    # Reshape to (n_batch*n_time_steps, n_conv_output_features)
    layers.append(lasagne.layers.ReshapeLayer(
        layers[-1], (-1, conv_output_shape[2]*conv_output_shape[3])))
    # Add dense hidden layers and optionally dropout
    for hidden_layer_size in hidden_layer_sizes:
        layers.append(lasagne.layers.DenseLayer(
            layers[-1], num_units=hidden_layer_size,
            nonlinearity=lasagne.nonlinearities.rectify))
        if dropout:
            layers.append(lasagne.layers.DropoutLayer(layers[-1], .5))
    # Add output layer
    layers.append(lasagne.layers.DenseLayer(
        layers[-1], num_units=n_bits,
        nonlinearity=lasagne.nonlinearities.tanh))

    return layers


def save_model(param_list, output_file):
    '''
    Write out a pickle file of a hashing network

    Parameters
    ----------
    param_list : list of np.ndarray
        A list of values, per layer, of the parameters of the network
    output_file : str
        Path to write the file to
    '''
    with open(output_file, 'wb') as f:
        pickle.dump(param_list, f)


def load_model(layers, param_file):
    '''
    Load in the parameters from a pkl file into a model

    Parameters
    ----------
    layers : list
        A list of layers which define the model
    param_file : str
        Pickle file of model parameters to load
    '''
    with open(param_file) as f:
        params = pickle.load(f)
    lasagne.layers.set_all_param_values(layers[-1], params)
