"""
loads the semi-structured BGL drain data, and processes it for the CNN
"""

import numpy as np
import pandas as pd
import time
from sliding_window_processor import collect_event_ids, FeatureExtractor

if __name__ == "__main__":

    # where the "raw" data for this file is located
    load_data_location = "../../project_processed_data/"

    # where the processed data is saved
    save_location = "../../project_processed_data/"

    start = time.time()

    # Loads data
    print("loading x_train")
    x_train = pd.read_csv("{}BGL_train.log_structured.csv".format(load_data_location))

    print("loading x_test")
    x_test = pd.read_csv("{}BGL_test.log_structured.csv".format(load_data_location))

    # print("loading y")
    # y = pd.read_csv("{}anomaly_label.csv".format(load_data_location))

    pd.set_option("display.max_columns", None)

    print(x_train.head())
    print(x_train.columns)
    print(x_train.shape)

    x_train_head = x_train.head(100000)

    print(x_train_head.shape)

    x_train_head.to_csv("{}x_train_head.csv".format(save_location))

    print("time taken: ", time.time() - start)
