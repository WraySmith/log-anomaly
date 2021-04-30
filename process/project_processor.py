"""
loads the semi-structured drain data, and processes it for the CNN
"""

import numpy as np
import pandas as pd
import time
from sliding_window_processor import collect_event_ids, FeatureExtractor

if __name__ == "__main__":

    data_version = "_v5"

    data_version = "_tf-idf{}".format(data_version)

    # where the "raw" data for this file is located
    load_data_location = "../../project_processed_data/"

    # where the processed data is saved
    save_location = "../../project_processed_data/{}/".format(data_version)

    start = time.time()

    # Loads data
    print("loading x_train")
    x_train = pd.read_csv("{}HDFS_train.log_structured.csv".format(load_data_location))

    print("loading x_test")
    x_test = pd.read_csv("{}HDFS_test.log_structured.csv".format(load_data_location))

    print("loading y")
    y = pd.read_csv("{}anomaly_label.csv".format(load_data_location))

    # processes events into blocks
    re_pat = r"(blk_-?\d+)"
    col_names = ["BlockId", "EventSequence"]

    print("collecting events for x_train")
    events_train = collect_event_ids(x_train, re_pat, col_names)
    print("collecting events for x_test")
    events_test = collect_event_ids(x_test, re_pat, col_names)

    print("merging block frames with labels")
    events_train = events_train.merge(y, on="BlockId")
    events_test = events_test.merge(y, on="BlockId")

    print("removing blocks that are overlapped into train and test")
    overlapping_blocks = np.intersect1d(events_train["BlockId"], events_test["BlockId"])
    events_train = events_train[~events_train["BlockId"].isin(overlapping_blocks)]
    events_test = events_test[~events_test["BlockId"].isin(overlapping_blocks)]

    events_train_values = events_train["EventSequence"].values
    events_test_values = events_test["EventSequence"].values

    # fit transform & transform
    fe = FeatureExtractor()

    print("fit_transform x_train")
    subblocks_train = fe.fit_transform(
        events_train_values,
        term_weighting="tf-idf",
        length_percentile=95,
        window_size=16,
    )

    print("transform x_test")
    subblocks_test = fe.transform(events_test_values)

    print("collecting y data")
    y_train = events_train[["BlockId", "Label"]]
    y_test = events_test[["BlockId", "Label"]]

    # saving files
    print("writing y to csv")
    y_train.to_csv("{}y_train{}.csv".format(save_location, data_version))
    y_test.to_csv("{}y_test{}.csv".format(save_location, data_version))

    print("saving x to numpy object")
    np.save("{}x_train{}.npy".format(save_location, data_version), subblocks_train)
    np.save("{}x_test{}.npy".format(save_location, data_version), subblocks_test)

    print("time taken :", time.time() - start)
