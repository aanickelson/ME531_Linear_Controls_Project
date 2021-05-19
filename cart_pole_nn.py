#!/usr/bin/env python3

import numpy as np
# from keras import models
# from keras import layers


if __name__ == "__main__":

    all_data = np.load("cart-pole-data-20210518-160517.npy")

    # inputs = all_data[:, :4]
    outputs = all_data[:, 4:5]
    inputs = np.delete(all_data, 4, axis=1)

    print(inputs)
    print(outputs)
