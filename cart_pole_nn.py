#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import time


def create_a_nn():
    network = tf.keras.Sequential()
    network.add(tf.keras.layers.Dense(6, input_shape=(4,)))
    network.add(tf.keras.layers.Dense(1))
    network.compile(optimizer='Adam', loss='mean_squared_error')
    return network


def train_the_nn(data_in, data_out):
    weights_path = 'weights-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=weights_path,
    #                                                  save_weights_only=True,
    #                                                  verbose=1)
    model = create_a_nn()
    model.fit(data_in, data_out, epochs=5)  #, callbacks=[cp_callback])
    model.save_weights(weights_path)


def use_trained_nn(weights_file):
    model = create_a_nn()
    model.load_weights(weights_file)
    return model


if __name__ == "__main__":

    all_data = np.load("cart-pole-data-20210521-123946.npy")
    print(all_data)

    # inputs = all_data[:, :4]
    all_out = all_data[:, 4:]
    all_in = np.delete(all_data, 4, axis=1)

    # print(all_in)
    # print(all_out)

    train_the_nn(all_in, all_out)

