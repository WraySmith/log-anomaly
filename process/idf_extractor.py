"""
loads the semi-structured BGL drain data, and processes it to get the idf
to be used when chunking
"""

import numpy as np
import pandas as pd
import time
from sliding_window_processor import collect_event_ids, FeatureExtractor, block_by_time

if __name__ == "__main__":

    chunk_size = 10000

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

    # time windows
    sliding_size = 3600
    window_size = 3600
    event_min_count = 25
    x_train, y_train = block_by_time(
        train_data, event_min_count, sliding_size, window_size
    )

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i: i + n]

    x_train_chunks = chunks(x_train, chunk_size)
    y_train_chunks = chunks(y_train, chunk_size)

    x_train_chunks = list(x_train_chunks)
    y_train_chunks = list(y_train_chunks)

    print("number of chunks", len(x_train_chunks))
    print("length insisde chunk", len(x_train_chunks[0]))

    # over rides for multiple chunks
    override_events = set(np.concatenate(x_train).ravel().flatten())
    length_list = np.array(list(map(len, x_train)))
    override_lenghts = int(
        np.percentile(length_list, 95))

    for i in range(len(x_train_chunks)):

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
