"""
loads and preprocesses the structured log data for anomaly prediction
"""
import numpy as np
import pandas as pd
import time
import re
from collections import OrderedDict
from collections import Counter


def collect_event_ids(data_frame, regex_pattern, column_names):
    """
    turns input data_frame into a 2 columned dataframe
    with columns: BlockId, EventSequence
    where EventSequence is a list of the events that happened to the block
    """
    data_dict = OrderedDict()
    for _, row in data_frame.iterrows():
        blk_id_list = re.findall(regex_pattern, row["Content"])
        blk_id_set = set(blk_id_list)
        for blk_id in blk_id_set:
            if blk_id not in data_dict:
                data_dict[blk_id] = []
            data_dict[blk_id].append(row["EventId"])
    data_df = pd.DataFrame(list(data_dict.items()), columns=column_names)
    return data_df


class FeatureExtractor(object):
    """
    class for fitting and transforming the training set
    then transforming the testing set
    """

    def __init__(self):
        self.mean_vec = None
        self.idf_vec = None
        self.events = None
        self.term_weighting = None

    def fit_transform(self, X_seq, term_weighting=None):
        """
        Fit and transform the training set
        X_Seq: ndarray,  log sequences matrix
        term_weighting: None or `tf-idf`
        normalization: None - not implemented yet
        """
        self.term_weighting = term_weighting

        # Convert into bag of words
        X_counts = []
        for i in range(X_seq.shape[0]):
            event_counts = Counter(X_seq[i])
            X_counts.append(event_counts)
        X_df = pd.DataFrame(X_counts)
        X_df = X_df.fillna(0)
        self.events = X_df.columns
        X = X_df.values

        num_instance, num_event = X.shape
        # applies tf-idf if pararmeter
        if self.term_weighting == "tf-idf":
            df_vec = np.sum(X > 0, axis=0)
            self.idf_vec = np.log(num_instance / (df_vec + 1e-8))
            idf_matrix = X * np.tile(self.idf_vec, (num_instance, 1))
            X = idf_matrix

        X_new = X
        print("Train data shape: {}-by-{}\n".format(X_new.shape[0], X_new.shape[1]))
        return X_new

    def fit_transform_subblocks(self, X_seq, term_weighting=None, rolling=False):
        """
        Fit and transform the training set
        X_Seq: ndarray,  log sequences matrix
        term_weighting: None or `tf-idf`
        rolling: Bool, default False, if True, applies a 2 window rolling sum to the dataframes
        """
        self.term_weighting = term_weighting
        self.rolling = rolling

        # get unique events
        unique_events = set()
        for i in X_seq:
            unique_events.update(i)
        self.events = unique_events

        # Convert into bag of words
        all_blocks_count = []
        for block in X_seq:
            # multiply block by 20 for 5% partitions
            block_rep = np.repeat(block, 20)
            # now split into 5% partitions
            block_split = np.split(block_rep, 20)
            block_counts = []
            for sub_block in block_split:
                # count each sub_block
                subset_count = Counter(sub_block)
                block_counts.append(subset_count)
            # put into dataframe to add nas to missing events
            # divide by 20 as original operation multiplied by 20
            block_df = pd.DataFrame(block_counts, columns=self.events) / 20
            block_df = block_df.reindex(sorted(block_df.columns), axis=1)
            block_df = block_df.fillna(0)
            if self.rolling:
                block_df = block_df.rolling(window=2).sum()
                block_df = block_df.dropna()

            block_np = block_df.to_numpy()
            all_blocks_count.append(block_np)

        # finally stack the blocks
        X = np.stack(all_blocks_count)

        # applies tf-idf if pararmeter
        if self.term_weighting == "tf-idf":

            # Set up sizing
            num_instance, _, _ = X.shape
            dim1, dim2, dim3 = X.shape
            X = X.reshape(-1, dim3)

            # apply tf-idf
            df_vec = np.sum(X > 0, axis=0)
            self.idf_vec = np.log(num_instance / (df_vec + 1e-8))
            idf_tile = np.tile(self.idf_vec, (num_instance * dim2, 1))
            idf_matrix = X * idf_tile
            X = idf_matrix

            # reshape to original dimensions
            X = X.reshape(dim1, dim2, dim3)

        X_new = X
        print("Train data shape: ", X_new.shape)
        return X_new

    def transform(self, X_seq):
        """
        transforms x test
        X_seq: log sequence data
        """

        # converts into bag of words
        X_counts = []
        for i in range(X_seq.shape[0]):
            event_counts = Counter(X_seq[i])
            X_counts.append(event_counts)
        X_df = pd.DataFrame(X_counts)
        X_df = X_df.fillna(0)
        empty_events = set(self.events) - set(X_df.columns)
        for event in empty_events:
            X_df[event] = [0] * len(X_df)
        X = X_df[self.events].values

        # applies tf-idf if parameter
        num_instance, _ = X.shape
        if self.term_weighting == "tf-idf":
            idf_matrix = X * np.tile(self.idf_vec, (num_instance, 1))
            X = idf_matrix

        X_new = X
        print("Test data shape: {}-by-{}\n".format(X_new.shape[0], X_new.shape[1]))
        return X_new

    def transform_subblocks(self, X_seq):
        """
        transforms x test
        X_seq: log sequence data
        """

        # Convert into bag of words
        all_blocks_count = []
        for block in X_seq:
            # multiply block by 20 for 5% partitions
            block_rep = np.repeat(block, 20)
            # now split into 5% partitions
            block_split = np.split(block_rep, 20)
            block_counts = []
            for sub_block in block_split:
                # count each sub_block
                subset_count = Counter(sub_block)
                block_counts.append(subset_count)
            # put into dataframe to add nas to missing events
            # divide by 20 as original operation multiplied by 20
            block_df = pd.DataFrame(block_counts, columns=self.events) / 20
            block_df = block_df.reindex(sorted(block_df.columns), axis=1)
            block_df = block_df.fillna(0)
            if self.rolling:
                block_df = block_df.rolling(window=2).sum()
                block_df = block_df.dropna()

            block_np = block_df.to_numpy()
            all_blocks_count.append(block_np)

        # finally stack the blocks
        X = np.stack(all_blocks_count)

        # applies tf-idf if parameter
        if self.term_weighting == "tf-idf":

            # set up sizing and reshape for tf-idf
            num_instance, _, _ = X.shape
            dim1, dim2, dim3 = X.shape
            X = X.reshape(-1, dim3)

            # applies tf-idf
            idf_tile = idf_tile = np.tile(self.idf_vec, (num_instance * dim2, 1))
            idf_matrix = X * idf_tile
            X = idf_matrix

            # reshape to original dimensions
            X = X.reshape(dim1, dim2, dim3)

        X_new = X
        print("Test data shape: ", X_new.shape)
        return X_new


if __name__ == "__main__":

    data_version = "_tf-idf_rolling_v2"

    # where the "raw" data for this file is located
    load_data_location = "../parse/project_parsed/"

    # where the processed data is saved
    save_location = "./project_processed_data/"

    start = time.time()

    print("loading y")
    y = pd.read_csv("{}anomaly_label.csv".format(load_data_location))

    print("loading x_train")
    x_train = pd.read_csv("{}HDFS_train.log_structured.csv".format(load_data_location))

    print("loading x_test")
    x_test = pd.read_csv("{}HDFS_test.log_structured.csv".format(load_data_location))

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
    subblocks_train = fe.fit_transform_subblocks(
        events_train_values, term_weighting="tf-idf", rolling=True
    )

    print("transform x_test")
    subblocks_test = fe.transform_subblocks(events_test_values)

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
