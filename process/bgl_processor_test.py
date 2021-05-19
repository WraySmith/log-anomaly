"""
loads the semi-structured BGL drain data, and processes it for the CNN
"""

import os
import numpy as np
import pandas as pd
import time
from sliding_window_processor import collect_event_ids, FeatureExtractor, block_by_time
import math

if __name__ == "__main__":

    min_event_count = 15

    # where the "raw" data for this file is located
    load_data_location = "../../project_processed_data/"

    # where the processed data is saved
    save_location = "../../project_processed_data/"

    window_size = 1
    step_size = 1

    para = {
        "save_path": save_location,
        "window_size": window_size,
        "sliding_size": step_size,
    }
    timer_start = time.time()

    # load the train templates

    # join with train data set

    # drop if less than desired threshold

    # Loads data
    # print("loading x_train")
    # x_train = pd.read_csv("{}x_train_head.csv".format(load_data_location))

    # print("first")
    # print(x_train.head(100))
    # print("last")
    # print(x_train.tail(1))

    # train_data = np.load("{}bgl_x_train.npy".format(load_data_location))
    # test_data = np.load("{}bgl_x_test.npy".format(load_data_location))

    y_train = pd.read_csv(
        "{}bgl_y_train.csv".format(load_data_location), index_col=None
    )
    y_test = pd.read_csv("{}bgl_y_test.csv".format(load_data_location))

    import statistics

    y_train = list(y_train.iloc[:, 1].values)
    y_test = list(y_test.iloc[:, 1].values)

    print(sum(y_train) / len(y_train))
    print(sum(y_test) / len(y_test))

    print("time taken: ", time.time() - timer_start)
