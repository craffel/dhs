'''
Functions for mapping data in different modalities to a common Hamming space
'''
import numpy as np
import theano.tensor as T
import theano
import lasagne
from . import utils
import collections


def train(data, layers, negative_importance, negative_threshold,
          entropy_importance, updates_function, batch_size=50,
          sequence_length=100, epoch_size=100, initial_patience=1000,
          improvement_threshold=0.99, patience_increase=10, max_iter=100000):
    '''
    Utility function for training a siamese network for (potentially
    cross-modal) hashing of sequences.
    Assumes X_train[n] should be mapped close to Y_train[m] only when n == m
    The networks for hashing sequences from each modality should be given in
    the ``layers`` dictionary (see below).

    Parameters
    ----------
    data : dict of dict of list of np.ndarray
        Dict with keys ``'X'`` and ``'Y'``, corresponding to each modality,
        with each key mapping to a dict with keys ``'train'`` and
        ``'validate'``, each of which containing a list of np.ndarrays of shape
        ``(n_filters, n_time_steps, n_features)``.
    layers : dict of list of lasagne.layers.Layer
        This should be a dict with two keys, ``'X'`` and ``'Y'``, with each key
        mapping to a list of ``lasagne.layers.Layer`` instance corresponding to
        the layers in each network.  The only constraints are that the input
        shape should match the shape produced by ``sample_sequences`` when it's
        called with the provided data arrays (``data['X']['train']``, etc.),
        that the output dimensionality of both networks should be the same, and
        that the output nonlinearity is tanh.
    negative_importance : float
        Scaling parameter for cross-modality negative example cost
    negative_threshold : int
        Cross-modality negative example threshold
    entropy_importance : float
        Scaling parameter for hash entropy encouraging term
    updates_function : function
        Function for computing updates, probably from ``lasagne.updates``.
        Should take two arguments, a Theano tensor variable and a list of
        shared variables, and should return a dictionary of updates for those
        parameters (all other arguments, such as learning rate, should be
        factored out).
    batch_size : int
        Mini-batch size
    sequence_length : int
        Size of extracted sequences
    epoch_size : int
        Number of mini-batches per epoch
    initial_patience : int
        Always train on at least this many batches
    improvement_threshold : float
        Validation cost must decrease by this factor to increase patience
    patience_increase : int
        How many more epochs should we wait when we increase patience
    max_iter : int
        Maximum number of batches to train on

    Returns
    -------
    epoch : iterator
        Results for each epoch are yielded
    '''
    # First neural net, for X modality
    X_p_input = T.tensor4('X_p_input')
    X_n_input = T.tensor4('X_n_input')
    # For eval
    X_input = T.tensor4('X_input')
    # Second neural net, for Y modality
    Y_p_input = T.tensor4('Y_p_input')
    Y_n_input = T.tensor4('Y_n_input')
    Y_input = T.tensor4('Y_input')

    # Compute \sum max(0, m - ||a - b||_2)^2
    def hinge_cost(m, a, b):
        dist = m - T.sqrt(T.sum((a - b)**2, axis=1))
        return T.mean((dist*(dist > 0))**2)

    def hasher_cost(deterministic):
        X_p_output = lasagne.layers.get_output(
            layers['X'][-1], X_p_input, deterministic=deterministic)
        X_n_output = lasagne.layers.get_output(
            layers['X'][-1], X_n_input, deterministic=deterministic)
        Y_p_output = lasagne.layers.get_output(
            layers['Y'][-1], Y_p_input, deterministic=deterministic)
        Y_n_output = lasagne.layers.get_output(
            layers['Y'][-1], Y_n_input, deterministic=deterministic)

        # Unthresholded, unscaled cost of positive examples across modalities
        cost_p = T.mean(T.sum((X_p_output - Y_p_output)**2, axis=1))
        # Thresholded, scaled cost of cross-modality negative examples
        cost_n = negative_importance*hinge_cost(
            negative_threshold, X_n_output, Y_n_output)
        # Cost to encourage each output unit to vary
        cost_e = entropy_importance*T.mean(
            T.sum(X_p_output**2, axis=1) + T.sum(Y_p_output**2, axis=1))
        # Sum positive and negative costs for overall cost
        cost = cost_p + cost_n + cost_e
        return cost

    # Combine all parameters from both networks
    params = (lasagne.layers.get_all_params(layers['X'][-1], trainable=True)
              + lasagne.layers.get_all_params(layers['Y'][-1], trainable=True))
    # Compute RMSProp gradient descent updates
    updates = updates_function(hasher_cost(False), params)
    # Function for training the network
    train = theano.function(
        [X_p_input, X_n_input, Y_p_input, Y_n_input], hasher_cost(False),
        updates=updates)

    # Compute cost without training
    cost = theano.function(
        [X_p_input, X_n_input, Y_p_input, Y_n_input], hasher_cost(True))

    # Start with infinite validate cost; we will always increase patience once
    current_validate_cost = np.inf
    patience = initial_patience

    # Functions for computing the neural net output on the train and val sets
    X_output = theano.function([X_input], lasagne.layers.get_output(
        layers['X'][-1], X_input, deterministic=True))
    Y_output = theano.function([Y_input], lasagne.layers.get_output(
        layers['Y'][-1], Y_input, deterministic=True))

    # Extract sample seqs from the validation set (only need to do this once)
    X_validate, Y_validate = utils.sample_sequences(
        data['X']['validate'], data['Y']['validate'], sequence_length)

    # Create fixed negative example validation set
    X_validate_shuffle = np.random.permutation(X_validate.shape[0])
    Y_validate_shuffle = X_validate_shuffle[
        utils.random_derangement(X_validate.shape[0])]
    X_validate_n = X_validate[X_validate_shuffle]
    Y_validate_n = Y_validate[Y_validate_shuffle]
    # We won't know the # of samples in X_val_output until after computing it,
    # so we will set this to None to mark it as needing later computing
    X_val_output_shuffle = None
    data_iterator = utils.get_next_batch(
        data['X']['train'], data['Y']['train'], batch_size, sequence_length,
        max_iter)
    # We will accumulate the mean train cost over each epoch
    train_cost = 0

    for n, (X_p, Y_p, X_n, Y_n) in enumerate(data_iterator):
        # Occasionally Theano was raising a MemoryError, this fails gracefully
        try:
            train_cost += train(X_p, X_n, Y_p, Y_n)
        except MemoryError:
            return
        # Stop training if a NaN is encountered
        if not np.isfinite(train_cost):
            print 'Bad training cost {} at iteration {}'.format(train_cost, n)
            break
        # Validate the net after each epoch
        if n and (not n % epoch_size):
            epoch_result = collections.OrderedDict()
            epoch_result['iteration'] = n
            # Compute average training cost over the epoch
            epoch_result['train_cost'] = train_cost / float(epoch_size)
            # Reset training cost mean accumulation
            train_cost = 0

            # We need to accumulate the validation cost and network output over
            # batches to avoid MemoryErrors
            epoch_result['validate_cost'] = 0
            validate_batches = 0
            X_val_output = []
            Y_val_output = []
            for batch_idx in range(0, X_validate.shape[0], batch_size):
                # Extract slice from validation set for this batch
                batch_slice = slice(batch_idx, batch_idx + batch_size)
                # Compute and accumulate cost
                epoch_result['validate_cost'] += cost(
                    X_validate[batch_slice], X_validate_n[batch_slice],
                    Y_validate[batch_slice], Y_validate_n[batch_slice])
                # Keep track of # of batches for normalization
                validate_batches += 1
                # Compute network output and accumulate result
                X_val_output.append(X_output(X_validate[batch_slice]))
                Y_val_output.append(Y_output(Y_validate[batch_slice]))
            # Normalize cost by number of batches and store
            epoch_result['validate_cost'] /= float(validate_batches)
            # Concatenate per-batch output to tensors
            X_val_output = np.concatenate(X_val_output, axis=0)
            Y_val_output = np.concatenate(Y_val_output, axis=0)
            # Create a fixed shuffling of X_val_output
            if X_val_output_shuffle is None:
                X_val_output_shuffle = utils.random_derangement(
                    X_val_output.shape[0])

            in_dist, in_mean, in_std = utils.statistics(
                X_val_output > 0, Y_val_output > 0)
            out_dist, out_mean, out_std = utils.statistics(
                X_val_output[X_val_output_shuffle] > 0, Y_val_output > 0)
            epoch_result['validate_accuracy'] = in_dist[0]
            epoch_result['validate_in_class_distance_mean'] = in_mean
            epoch_result['validate_in_class_distance_std'] = in_std
            epoch_result['validate_collisions'] = out_dist[0]
            epoch_result['validate_out_of_class_distance_mean'] = out_mean
            epoch_result['validate_out_of_class_distance_std'] = out_std
            X_entropy = utils.hash_entropy(X_val_output > 0)
            epoch_result['validate_hash_entropy_X'] = X_entropy
            Y_entropy = utils.hash_entropy(Y_val_output > 0)
            epoch_result['validate_hash_entropy_Y'] = Y_entropy
            # Objective is bhattacharyya distance
            bhatt_coeff = np.sum(np.sqrt(in_dist*out_dist))
            epoch_result['validate_objective'] = bhatt_coeff

            if epoch_result['validate_cost'] < current_validate_cost:
                patience_cost = improvement_threshold*current_validate_cost
                if epoch_result['validate_cost'] < patience_cost:
                    patience += epoch_size*patience_increase
                current_validate_cost = epoch_result['validate_cost']

            # Yield scores and statistics for this epoch
            yield epoch_result

            if n > patience:
                break

    return
