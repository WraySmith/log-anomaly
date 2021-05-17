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
    print("loading x_train")
    x_train = pd.read_csv("{}x_train_head.csv".format(load_data_location))

    print("first")
    print(x_train.head(100))
    print("last")
    print(x_train.tail(1))

    ##########################################
    # TIME WINDOWING FUNCTION STARTS HERE
    ##########################################

    df = x_train.copy(deep=True)
    X, y = block_by_time(df, 25, 3600, 3600)

    # event_id_minimum_count = 25

    # start = df["Timestamp"].values[0]
    # stop = df["Timestamp"].values[-1]

    # # Drop rare events
    # event_id_count = df.groupby(["EventId"])["Timestamp"].describe()[["count"]]
    # event_id_filtered = event_id_count[event_id_count["count"] > event_id_minimum_count]
    # event_id_filtered = set(event_id_filtered.index.values)
    # df = df[df["EventId"].isin(event_id_filtered)]

    # # Create X and y by windowing on time
    # slide_size = 3600
    # window_size = 3600
    # time_block_labels = []
    # X = []
    # num_time_windows = math.ceil((stop - start) / slide_size)
    # for i in range(0, num_time_windows + 1):
    #     time_subset = df[
    #         (df["Timestamp"] >= start) & (df["Timestamp"] < start + window_size)
    #     ]

    #     # Collect the y's
    #     is_anom_labels = set(time_subset["Label"].unique()) - set(["-"])
    #     if is_anom_labels:
    #         is_anomalous = 1
    #     else:
    #         is_anomalous = 0
    #     time_block_labels.append(is_anomalous)

    #     # Collect the X's
    #     X.append(time_subset["EventId"].values)

    #     # Increment window start
    #     start += slide_size

    # assumes data has been sorted by time

    print("time taken: ", time.time() - timer_start)
