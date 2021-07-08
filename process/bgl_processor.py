"""
loads the semi-structured BGL drain data, and processes it for the CNN
"""

import numpy as np
import pandas as pd
import time
from sliding_window_processor import collect_event_ids, FeatureExtractor, block_by_time

if __name__ == "__main__":

    save_name_extention = "overlap_half_hour"

    # where the "raw" data for this file is located
    load_data_location = "../../project_processed_data/"

    # where the processed data is saved
    save_location = "../../project_processed_data/"

    start = time.time()

    # Loads data
    print("loading x_train")
    train_data = pd.read_csv(
        "{}BGL_train.log_structured.csv".format(load_data_location)
    )
    print("loading x_test")
    test_data = pd.read_csv("{}BGL_test.log_structured.csv".format(load_data_location))

    # time windows
    sliding_size = int(3600 / 2)
    window_size = 3600
    event_min_count = 25
    x_train, y_train = block_by_time(
        train_data, event_min_count, sliding_size, window_size
    )
    x_test, y_test = block_by_time(
        test_data, event_min_count, sliding_size, window_size
    )

    # fit transform & transform
    fe = FeatureExtractor()
    print("fit_transform x_train")
    subblocks_train = fe.fit_transform(
        x_train,
        term_weighting="tf-idf",
        length_percentile=95,
        window_size=16,
    )
    print("transform x_test")
    subblocks_test = fe.transform(x_test)

    # saving files
    print("writing y to csv")
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)

    y_train.to_csv("{}bgl_y_train_{}.csv".format(save_location, save_name_extention))
    y_test.to_csv("{}bgl_y_test_{}.csv".format(save_location, save_name_extention))

    print("saving x to numpy object")
    np.save(
        "{}bgl_x_train_{}.npy".format(save_location, save_name_extention),
        subblocks_train,
    )
    np.save(
        "{}bgl_x_test_{}.npy".format(save_location, save_name_extention), subblocks_test
    )

    print("time taken :", time.time() - start)
