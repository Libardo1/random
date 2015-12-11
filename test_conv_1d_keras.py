from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution1D
from network_fullyconnected import load_data
from keras.layers.convolutional import MaxPooling1D
from keras.layers.core import Flatten
from keras.optimizers import SGD  # , RMSprop
import keras.callbacks as cb
import matplotlib.pyplot as plt
from keras.regularizers import l2
import time


class LossHistory(cb.Callback):

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

print "loading ..."
x_train, x_test, y_train, y_test = load_data('timeseries', True, False, 'raw')


def conv_run():
    start = time.time()
    mod = Sequential()
    # --- --- --- --- --- --- --- --- --- --- ---
    mod.add(
        Convolution1D(10, 30, input_shape=(7500, 1), init='he_normal',
                      W_regularizer=l2(0.1))
    )
    mod.add(
        Activation('tanh')
    )
    mod.add(
        Dropout(0.3)
    )
    # --- --- --- --- --- --- --- --- --- --- ---
    # mod.add(
    #     Convolution1D(20, 30, init='he_normal',
    #                   W_regularizer=l2(0.01))
    # )
    # mod.add(
    #     Activation('tanh')
    # )
    # mod.add(
    #     Dropout(0.3)
    # )
    # --- --- --- --- --- --- --- --- --- --- ---
    mod.add(
        MaxPooling1D(pool_length=2, stride=1)
    )
    mod.add(
        Dropout(0.3)
    )
    # --- --- --- --- --- --- --- --- --- --- ---
    mod.add(
        Flatten()
    )
    # --- --- --- --- --- --- --- --- --- --- ---
    mod.add(
        Dense(1500, W_regularizer=l2(0.01))
    )
    mod.add(
        Activation('tanh')
    )
    mod.add(
        Dropout(0.3)
    )
    # --- --- --- --- --- --- --- --- --- --- ---
    mod.add(
        Dense(250, W_regularizer=l2(0.01))
    )
    mod.add(
        Activation('relu')
    )
    mod.add(
        Dropout(0.3)
    )
    # # --- --- --- --- --- --- --- --- --- --- ---
    # mod.add(
    #     Dense(30, W_regularizer=l2(0.1))
    # )
    # mod.add(
    #     Activation('tanh')
    # )
    # mod.add(
    #     Dropout(0.3)
    # )
    # --- --- --- --- --- --- --- --- --- --- ---
    # mod.add(
    #     Dense(2, init='he_normal')
    # )
    # mod.add(
    #     Activation('softmax')
    # )

    # --- --- --- --- --- --- --- --- --- --- ---
    print 'compiling ...'
    mod.compile(
        loss='mean_squared_error',
        optimizer=SGD(lr=1, decay=1e-6, momentum=0.9, nesterov=True)
    )

    callback = cb.EarlyStopping(
            monitor='val_loss', patience=50, verbose=1)

    history = LossHistory()

    print "compiled."

    mod.fit(
        x_train, y_train, nb_epoch=50, batch_size=1024,
        show_accuracy=True, verbose=2,
        validation_split=0.1,
        callbacks=[callback, history]
    )

    print mod.evaluate(
        x_test, y_test, batch_size=128, show_accuracy=True
    )

    print '\n training duration : ' + str(time.time()-start)

    return history.losses


def conv_plt(losses):
    for loss in losses:
        plt.plot(loss)
    plt.show()
