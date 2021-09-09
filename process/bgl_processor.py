"""
loads the semi-structured BGL drain data, and processes it for the CNN
"""

import numpy as np
import pandas as pd
import time
from sliding_window_processor import collect_event_ids, FeatureExtractor, block_by_time

if __name__ == "__main__":

    chunk_size = 1600

    save_name_extention = "overlap_half_hour_no_tfidf"

    # where the "raw" data for this file is located
    load_data_location = "../../project_processed_data/"

    # where the processed data is saved
    save_location = "../../project_processed_data/"

    start = time.time()

    import os
    cwd = os.getcwd()

    # Loads data
    print("loading x_train")
    train_data = pd.read_csv(
        "{}BGL_train.log_structured.csv".format(load_data_location)
    )
    print("loading x_test")
    test_data = pd.read_csv(
        "{}BGL_test.log_structured.csv".format(load_data_location))

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
            yield lst[i: i + n]

    print("size of full x train", len(x_train))

    x_train_chunks = chunks(x_train, chunk_size)
    y_train_chunks = chunks(y_train, chunk_size)

    x_train_chunks = list(x_train_chunks)
    y_train_chunks = list(y_train_chunks)

    print("number of chunks", len(x_train_chunks))
    print("length insisde chunk", len(x_train_chunks[0]))

    print("size of x_test", len(x_test))

    # over rides for multiple chunks
    override_events = set(np.concatenate(x_train).ravel().flatten())
    length_list = np.array(list(map(len, x_train)))
    override_lenghts = int(
        np.percentile(length_list, 95))

    for i in range(len(x_train_chunks)):

        # logging
        f = open("out_side_for_loop_chunks.txt", "w")
        f.write(f"\n chunk {i}")
        f.close()

        print("proccessing block {}".format(i))
        x_chunk = x_train_chunks[i]
        y_chunk = y_train_chunks[i]

        fe = FeatureExtractor()
        print("fit_transform x_train")
        subblocks_train = fe.fit_transform(
            x_chunk,
            term_weighting="tf-idf",
            length_percentile=95,
            window_size=16,
            max_seq_length_override=override_lenghts,
            events_override=override_events
        )

        exit()

        y_chunk = pd.Series(y_chunk)
        y_chunk.to_csv(
            "{}bgl_y_train_{}_{}.csv".format(
                save_location, save_name_extention, i)
        )
        np.save(
            "{}bgl_x_train_{}_{}.npy".format(
                save_location, save_name_extention, i),
            subblocks_train,
        )

    # TEST SET
    x_test_chunks = chunks(x_test, chunk_size)
    y_test_chunks = chunks(y_test, chunk_size)

    x_test_chunks = list(x_test_chunks)
    y_test_chunks = list(y_test_chunks)

    print("number of chunks", len(x_test_chunks))
    print("length insisde chunk", len(x_test_chunks[0]))

    print("size of x_test", len(x_test))

    print("transform x_test")
    subblocks_test = fe.transform(x_test)
    y_test = pd.Series(y_test)
    y_test.to_csv("{}bgl_y_test_{}.csv".format(
        save_location, save_name_extention))
    np.save(
        "{}bgl_x_test_{}.npy".format(
            save_location, save_name_extention), subblocks_test
    )

    for i in range(len(x_test_chunks)):

        # logging
        f = open("out_side_for_loop_chunks.txt", "w")
        f.write(f"\n chunk {i}")
        f.close()

        print("proccessing block {}".format(i))
        x_chunk = x_test_chunks[i]
        y_chunk = y_test_chunks[i]

        fe = FeatureExtractor()
        print("fit_transform x_test")
        subblocks_test = fe.fit_transform(
            x_chunk,
            term_weighting=None,
            length_percentile=95,
            window_size=16,
            max_seq_length_override=override_lenghts,
            events_override=override_events
        )

        y_chunk = pd.Series(y_chunk)
        y_chunk.to_csv(
            "{}bgl_y_test_{}_{}.csv".format(
                save_location, save_name_extention, i)
        )
        np.save(
            "{}bgl_x_test_{}_{}.npy".format(
                save_location, save_name_extention, i),
            subblocks_train,
        )

    # # fit transform & transform
    # fe = FeatureExtractor()
    # print("fit_transform x_train")
    # subblocks_train = fe.fit_transform(
    #     x_train,
    #     term_weighting=None,
    #     length_percentile=95,
    #     window_size=16,
    # )
    # print("transform x_test")
    # subblocks_test = fe.transform(x_test)

    # # saving files
    # print("writing y to csv")

    # # save the train

    # # save the test
    # y_test = pd.Series(y_test)
    # y_test.to_csv("{}bgl_y_test_{}.csv".format(
    #     save_location, save_name_extention))
    # np.save(
    #     "{}bgl_x_test_{}.npy".format(
    #         save_location, save_name_extention), subblocks_test
    # )
