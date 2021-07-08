"""
splits files for easier transfer
"""

import numpy as np
import time

if __name__ == "__main__":

    start = time.time()

    # where the "raw" data for this file is located
    load_data_location = "../../project_processed_data/"

    # where the processed data is saved
    save_location = "../../project_processed_data/"

    # Loads data
    print("loading x_train")
    train_data = np.load("{}bgl_x_train.npy".format(load_data_location))

    print("train shape", train_data.shape)

    s = 0
    np_chunks = np.array_split(train_data, 4)
    counter = 0
    for c in np_chunks:
        np.save("{}bgl_x_train_{}.npy".format(save_location, counter), c)
        counter += 1

    print("time taken :", time.time() - start)
