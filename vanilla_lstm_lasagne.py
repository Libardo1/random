import numpy as np
import theano
import theano.tensor as T
import time
import lasagne
import os
from lasagne.layers import *
import matplotlib.pyplot as plt
import pickle
import math
from lasagne.regularization import regularize_layer_params, l2
import wash_out_lstm as wol


global_path = '/home/valentin/data/victor/Dreem-Mathematics/'
global_path += 'dreem_mathematics/offline/victor_neural/dreem_networks/'

data_path = global_path + 'keras_dreem/'
data_path += 'timeseries_data/2015-11-30_10-23-35_data_657_11/'

global_path += 'lasagne_dreem/'


def iterate_minibatches(batch_size, M_train):

    nb = M_train.shape[0]
    for i in range(nb/batch_size):
        temp_M = M_train[i*batch_size:(i+1)*batch_size, :]
        yield temp_M


def build_mlp(size_x, lstm_size, input_var=None):

    lstm_nonlinearity = lasagne.nonlinearities.sigmoid
    gate_parameters = lasagne.layers.recurrent.Gate(
            W_in=lasagne.init.Orthogonal(),
            W_hid=lasagne.init.Orthogonal(),
            b=lasagne.init.Constant(0.))
    cell_parameters = lasagne.layers.recurrent.Gate(
            W_in=lasagne.init.Orthogonal(),
            W_hid=lasagne.init.Orthogonal(),
            # Setting W_cell to None denotes that no
            # cell connection will be used.
            W_cell=None,
            b=lasagne.init.Constant(0.),
            # By convention, the cell nonlinearity is tanh in an LSTM.
            nonlinearity=lasagne.nonlinearities.tanh)

    l_in = InputLayer((None, None, size_x),
                      input_var=input_var)
    batch_size, seqlen, _ = l_in.input_var.shape
    l_lstm = LSTMLayer(l_in, lstm_size, learn_init=True,
                       nonlinearity=lstm_nonlinearity,
                       ingate=gate_parameters,
                       forgetgate=gate_parameters,
                       cell=cell_parameters,
                       outgate=gate_parameters,
                       grad_clipping=100.)
    l2_penalty = regularize_layer_params(l_lstm, l2)
    l_reshape = lasagne.layers.ReshapeLayer(l_lstm, (-1, lstm_size))
    # Now, we can apply feed-forward layers as usual.
    l_dense = lasagne.layers.DenseLayer(
     l_reshape, num_units=1, nonlinearity=None)
    # Now, the shape will be n_batch*n_timesteps, 1. We can then reshape to
    # batch_size, seqlen to get a single value
    # for each timstep from each sequence
    l_out = lasagne.layers.ReshapeLayer(l_dense, (batch_size, seqlen, size_x))
    # l1_penalty = regularize_layer_params(l_out, l2)
    return l_out, l2_penalty  # , l1_penalty


def load_data(train, r=0.1):
    ls = wol.Lstm(10, 1)
    if train:
        M = ls.load_ts_data(r0=0, r1=r, verbose=0)
    else:
        M = ls.load_ts_data(r0=0, r1=r, verbose=0, returning='test')
    print M.size
    return M


def run():
    hidden_size = 20
    nb_epoch = 10
    batch_size = 100
    input_size = 1
    learning_rate = 1e-2
    signal_size = 11*250
    optimizer = 'rms'

    train_batches = 0
    test_batches = 0
    train_err = 0
    test_err = 0

    input_var = T.tensor3('inputs')
    target_var = T.tensor3('targets')

    network, l2_penalty = build_mlp(
        input_size, hidden_size, input_var=input_var)
    prediction = lasagne.layers.get_output(network)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    cost = T.mean((prediction - target_var)**2) + l2_penalty
    test_cost = T.mean((test_prediction - target_var)**2)

    params = lasagne.layers.get_all_params(network)
    if optimizer == 'nesterov_momentum':
        updates = lasagne.updates.nesterov_momentum(
            cost, params, learning_rate=learning_rate, momentum=0.9)
    else:
        updates = lasagne.updates.rmsprop(
            cost, params, learning_rate=learning_rate)
    print 'compiling'
    train_fn = theano.function([input_var, target_var], cost, updates=updates)
    test_fn = theano.function([input_var, target_var], test_cost)

    print 'loading M_test:'
    M_test = load_data(False, 0.001).copy()
    try:
        for epoch in range(nb_epoch):
            epoch += 1
            print '\nepoch {}'.format(epoch)
            M_train = load_data(True, 0.001).copy()
            M_train[M_train > 500] = 500
            M_train[M_train < -500] = -500
            M_train /= 34.145035
            if M_train.shape[1] > signal_size:
                M_train = M_train[:, :signal_size]
            if batch_size > M_train.shape[0]:
                batch_size = M_train.shape[0]

            gen_batch = iterate_minibatches(batch_size, M_train)
            for batch in gen_batch:
                train_batches += signal_size * batch_size
                print 'Training {} / {}'.format(
                    train_batches, epoch * M_train.size)
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                # numpy new axis, copy
                inputs = np.reshape(
                                inputs, (inputs.shape[0],
                                         inputs.shape[1],
                                         1))
                targets = np.reshape(
                                targets, (targets.shape[0],
                                          targets.shape[1],
                                          1))
                train_err += train_fn(inputs, targets)

            test_gen_batch = iterate_minibatches(batch_size, M_test)
            for test_batch in test_gen_batch:
                test_batches += signal_size * batch_size
                print 'Testing {} / {}'.format(test_batches, M_test.size)
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                inputs = np.reshape(
                                inputs, (inputs.shape[0],
                                         inputs.shape[1],
                                         1))
                targets = np.reshape(
                                targets, (targets.shape[0],
                                          targets.shape[1],
                                          1))
                test_err += test_fn(inputs, targets)

            print "\ntraining loss: {}".format(train_err / train_batches)
            print "validation loss: {}".format(test_err / test_batches)
    except KeyboardInterrupt:
        return network

    return network

#
#
#
#
#
