"""
Loads x_train and y_train

then splits into train and val sets
"""

import pandas as pd
import numpy as np
import random

if __name__ == "__main__":

    # enter the size of the val set as a percentage of the train set
    val_percent_size = 0.10

    data_location = "../../project_processed_data/"

    x_train = np.load("{}x_train_tf-idf_rolling_v2.npy".format(data_location))
    y_train = pd.read_csv("{}y_train_tf-idf_rolling_v2.csv".format(data_location))

    old_x_train_shape = x_train.shape
    old_y_train_shape = y_train.shape
    print("x_train shape")
    print(x_train.shape)
    print("y_train shape")
    print(y_train.shape)

    assert x_train.shape[0] == y_train.shape[0]

    random.seed(10)

    val_size = int(x_train.shape[0] * val_percent_size)
    val_idx = random.sample(range(0, x_train.shape[0]), val_size)

    for item in val_idx:
        assert item >= 0
        assert item <= x_train.shape[0]

    x_val = np.take(x_train, val_idx, axis=0)
    print("x_val shape")
    print(x_val.shape)
    y_val = y_train.iloc[val_idx]
    print("y_val shape")
    print(y_val.shape)

    assert y_val.shape[0] == x_val.shape[0]

    assert y_val.iloc[0].equals(y_train.iloc[val_idx[0]])
    assert np.array_equal(x_val[0], x_train[val_idx[0]])

    anomalies = y_val[y_val["Label"] == "Anomaly"]
    print("number of anomalies in val set", anomalies.shape)
    print(
        "percentage of records that are anonylous in val set",
        anomalies.shape[0] / y_val.shape[0],
    )

    # delete from x_train, y_train the val_idx
    x_train = np.delete(x_train, val_idx, axis=0)
    y_train = y_train.drop(y_train.index[[val_idx]])

    print(x_train.shape[0], x_val.shape[0], old_x_train_shape[0])
    print(y_train.shape[0], y_val.shape[0], old_y_train_shape[0])
    assert x_train.shape[0] + x_val.shape[0] == old_x_train_shape[0]
    assert y_train.shape[0] + y_val.shape[0] == old_y_train_shape[0]

    print("loaded, split, and checked")
    np.save("{}x_val_tf-idf_rolling_v2.npy".format(data_location), x_val)
    y_val.to_csv("{}y_val_tf-idf_rolling_v2.csv".format(data_location))
    print("data saved")
