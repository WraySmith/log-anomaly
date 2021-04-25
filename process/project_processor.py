"""
loads the semi-structured drain data, and processes it for the CNN
"""

import numpy as np
import pandas as pd
import time
from data_processor_v2 import collect_event_ids, FeatureExtractor
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    testing = True

    data_version = "_tf-idf_rolling_v4"

    # where the "raw" data for this file is located
    load_data_location = "../../project_processed_data/"

    # where the processed data is saved
    save_location = "../../project_processed_data/v4/"

    start = time.time()

    # print("loading data")
    # main_data = pd.read_csv("{}HDFS.log_structured.csv".format(load_data_location))
    # print(main_data.head())
    # print(main_data.shape)

    # print("loading y")
    # y = pd.read_csv("{}anomaly_label.csv".format(load_data_location))

    # # for testing
    # if testing:
    #     main_data = main_data.head(1000)

    # re_pat = r"(blk_-?\d+)"
    # col_names = ["BlockId", "EventSequence"]

    # print("collecting events for the whole data set")
    # events_frame = collect_event_ids(main_data, re_pat, col_names)

    # # for testing
    # if testing:
    #     y = y.head(events_frame.shape[0])

    # print("events frame shape : ", events_frame.shape)
    # print("y frame shape : ", y.shape)
    # print("merge x and y into one data frame")

    # events_frame = events_frame.merge(y, on="BlockId")

    # print(events_frame.head())

    # train, test = train_test_split(events_frame, test_size=0.4)

    # train, val = train_test_split(train, test_size=0.17)

    # print("train shape")
    # print(train.head())
    # print(train.shape)
    # print("test shape")
    # print(test.head())
    # print(test.shape)
    # print("val shape")
    # print(val.head())
    # print(val.shape)

    # print("fitting and transforming")

    # fe = FeatureExtractor()

    # print("fit_transform x_train")
    # subblocks_train = fe.fit_transform_subblocks(
    #     train["EventSequence"].values, term_weighting="tf-idf", rolling=True
    # )

    # print("transform x_test")
    # subblocks_test = fe.transform_subblocks(test["EventSequence"].values)

    # print(subblocks_train.shape)
    # print(subblocks_test.shape)
    # print(subblocks_val.shape)

    # y_train = train[["BlockId", "Label"]]
    # y_test = test[["BlockId", "Label"]]
    # y_val = val[["BlockId", "Label"]]

    # print("y train")
    # print(y_train.head())
    # print("y test")
    # print(y_test.head())
    # print("y val")
    # print(y_val.head())

    # print("saving files")
    # print("writing y to csv")
    # y_train.to_csv("{}y_train{}.csv".format(save_location, data_version))
    # y_test.to_csv("{}y_test{}.csv".format(save_location, data_version))
    # y_val.to_csv("{}y_val{}.csv".format(save_location, data_version))

    # print("saving x to numpy object")
    # np.save("{}x_train{}.npy".format(save_location, data_version), subblocks_train)
    # np.save("{}x_test{}.npy".format(save_location, data_version), subblocks_test)
    # np.save("{}x_val{}.npy".format(save_location, data_version), subblocks_val)

    print("loading x_train")
    x_train = pd.read_csv("{}HDFS_train.log_structured.csv".format(load_data_location))

    print("loading x_test")
    x_test = pd.read_csv("{}HDFS_test.log_structured.csv".format(load_data_location))

    print("loading y")
    y = pd.read_csv("{}anomaly_label.csv".format(load_data_location))

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

    fe = FeatureExtractor()

    print("fit_transform x_train")
    subblocks_train = fe.fit_transform(
        events_train_values, term_weighting="tf-idf", length_percentile=100
    )

    print("transform x_test")
    subblocks_test = fe.fit_transform(events_test_values)

    print("collecting y data")
    y_train = events_train[["BlockId", "Label"]]
    y_test = events_test[["BlockId", "Label"]]

    print("writing y to csv")
    y_train.to_csv("{}y_train{}.csv".format(save_location, data_version))
    y_test.to_csv("{}y_test{}.csv".format(save_location, data_version))

    print("saving x to numpy object")
    np.save("{}x_train{}.npy".format(save_location, data_version), subblocks_train)
    np.save("{}x_test{}.npy".format(save_location, data_version), subblocks_test)

    print("time taken :", time.time() - start)
