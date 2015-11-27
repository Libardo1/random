class Lstm():
    def __init__(self, x_duration, y_duration):
        self.x_duration = x_duration
        self.y_duration = y_duration

    def load_ts_data():
        # [...]
        return

    def iterate_minibatches(self, batch_size, x, y):

        nb = x.shape[0]
        for i in range(nb/batch_size):
            temp_x = x[i*batch_size:(i+1)*batch_size, :, :]
            temp_y = y[i*batch_size:(i+1)*batch_size, :]
            yield temp_x, temp_y

    def build_mlp(self, input_var=None):
        num_classes = self.params['y_duration']*250
        lstm_size = 100
        lstm_nonlinearity = lasagne.nonlinearities.tanh

        l_in = InputLayer((None, 10, 250),
                          input_var=input_var)
        batch_size, seqlen, _ = l_in.input_var.shape
        forget_gate = Gate(b=lasagne.init.Constant(5.0))
        l_lstm = LSTMLayer(l_in, lstm_size,
                           nonlinearity=lstm_nonlinearity,
                           cell_init=lasagne.init.GlorotNormal(),
                           hid_init=lasagne.init.GlorotNormal(),
                           forgetgate=forget_gate)
        l_lstm_drop = DropoutLayer(l_lstm, p=0.4)
        l_forward_slice = SliceLayer(l_lstm_drop, -1, 1)
        l_out = DenseLayer(
                    l_forward_slice, num_units=num_classes, nonlinearity=None)

        return l_out


def run_lstm(x_duration, y_duration, learning_rate=0.01, num_epochs=30):
    global_start_time = time.time()
    # General parameters
    learning_rate = 0.00000001
    batch_size = 128
    test_batch_size = 400
    nb_epoch = 1
    # counters initialization
    train_err = 0
    train_batches = 0

    input_var = T.tensor3('inputs')
    target_var = T.fmatrix('targets')

    # --- --- --- --- --- --- ---

    ls = Lstm(x_duration, y_duration)
    network = ls.build_mlp(input_var)

    prediction = lasagne.layers.get_output(network)
    prediction = theano.printing.Print('PREDICTION : ')(prediction)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=learning_rate, momentum=0.9)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction,
                                                 target_var)
    test_loss = test_loss.mean()

    print ' === === === ===  C o m p i l a t i o n  === === === ===  '
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], test_loss)

    print "\n === === === === T r a i n i n g === === === === \n"
    examples = 0

    x_test, y_test = ls.load_ts_data()
    for epoch in range(nb_epoch):
        x_train, y_train = ls.load_ts_data()

        examples += x_train.shape[0]

        # In each epoch, we do a full pass over the training data:

        gen_batch = ls.iterate_minibatches(batch_size, x_train, y_train)
        for batch in gen_batch:
            inputs, targets = batch
            inputs = np.reshape(inputs, (batch_size, x_duration, 250))
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0

        gen_batch = ls.iterate_minibatches(test_batch_size, x_test, y_test)
        for batch in gen_batch:
            inputs, targets = batch
            inputs = np.reshape(inputs, (test_batch_size, x_duration, 250))
            err = val_fn(inputs, targets)
            val_err += err
            val_batches += 1

        print "training loss: {}".format(train_err / train_batches)
        print "validation loss: {}".format(val_err / val_batches)
