"""
loads the semi-structured BGL drain data, and processes it for the CNN
"""

import numpy as np
import pandas as pd
import time
from sliding_window_processor import collect_event_ids, FeatureExtractor, block_by_time

if __name__ == "__main__":

    chunk_size = 400

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

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    x_train_chunks = chunks(x_train, chunk_size)
    y_train_chunks = chunks(y_train, chunk_size)

    x_train_chunks = list(x_train_chunks)
    y_train_chunks = list(y_train_chunks)

    for i in range(len(x_train_chunks)):
        print("proccessing block {}".format(i))
        x_chunk = x_train_chunks[i]
        y_chunk = y_train_chunks[i]

        fe = FeatureExtractor()
        print("fit_transform x_train")
        subblocks_train = fe.fit_transform(
            x_train,
            term_weighting="tf-idf",
            length_percentile=95,
            window_size=16,
        )

        y_chunk = pd.Series(y_chunk)
        y_chunk.to_csv(
            "{}bgl_y_train_{}_{}.csv".format(save_location, save_name_extention, i)
        )
        np.save(
            "{}bgl_x_train_{}_{}.npy".format(save_location, save_name_extention, i),
            subblocks_train,
        )

    print("transform x_test")
    subblocks_test = fe.transform(x_test)
    y_test = pd.Series(y_test)
    y_test.to_csv("{}bgl_y_test_{}.csv".format(save_location, save_name_extention))
    np.save(
        "{}bgl_x_test_{}.npy".format(save_location, save_name_extention), subblocks_test
    )

    exit()

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

    # save the train

    # save the test
    y_test = pd.Series(y_test)
    y_test.to_csv("{}bgl_y_test_{}.csv".format(save_location, save_name_extention))
    np.save(
        "{}bgl_x_test_{}.npy".format(save_location, save_name_extention), subblocks_test
    )

    print("time taken :", time.time() - start)
