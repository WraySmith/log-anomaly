"""
loads the semi-structured BGL drain data, and processes it for the CNN
"""

import os
import numpy as np
import pandas as pd
import time
from sliding_window_processor import collect_event_ids, FeatureExtractor
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
    start = time.time()

    # load the train templates

    # join with train data set

    # drop if less than desired threshold

    # Loads data
    print("loading x_train")
    x_train = pd.read_csv("{}x_train_head.csv".format(load_data_location))

    print("first")
    print(x_train.head(100))
    print("last")
    print(x_train.tail(1))

    df = x_train.copy(deep=True)
    start = df["Timestamp"].values[0]
    stop = df["Timestamp"].values[-1]

    slide_size = 3600
    window_size = 3600
    sums = 0
    time_block_labels = []
    anomalous_counter = 0
    num_time_windows = math.ceil((stop - start) / slide_size)
    for i in range(0, num_time_windows + 1):
        time_subset = df[
            (df["Timestamp"] >= start) & (df["Timestamp"] < start + window_size)
        ]
        sums += len(time_subset)
        is_anom_labels = set(time_subset["Label"].unique()) - set(["-"])
        if is_anom_labels:
            print("anomalous")
            anomalous_counter += 1
        print(is_anom_labels)
        # exit()
        # if ():
        #     # if (
        #     #     len(time_subset[time_subset["Label"] == "-"])
        #     #     - len(time_subset[time_subset["Label"] != "-"])
        #     #     > 0
        #     # ):
        #     anomalous_counter += 1
        #     print("anomalyous")
        start += slide_size

    print(sums)
    print(anomalous_counter)
    exit(0)

    # windows rows based on time value

    # get first time stamp, caluclate + window size
    # this becomes our first "block"
    # all collect all events in range start & end

    # first data point in new smaller dataframe becomes start
    # etc...

    # assumes data has been sorted by time

    # actually can start at start time and calculate hours until end time
    # a priori

    print("time taken: ", time.time() - start)
