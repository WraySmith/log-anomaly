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

    def fit_transform_subblocks(self, X_seq, term_weighting=None):
        """
        Fit and transform the training set
        X_Seq: ndarray,  log sequences matrix
        term_weighting: None or `tf-idf`
        normalization: None - not implemented yet
        """
        self.term_weighting = term_weighting

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
            block_np = block_df.to_numpy()
            all_blocks_count.append(block_np)

        # finally stack the blocks
        X = np.stack(all_blocks_count)

        # CODE BELOW HERE IS NOT UPDATED
        # NEED TO 1: GET DOCUMENT COUNT (COULD POTENTIALLY BE INCORPORATED IN THE ABOVE BLOCK)
        # 2: APPLY THE SLIDE WINDOW SUCH THAT THE FINAL RESULT IS 20 ROWS OF 0-10, 5-15, 10-25
        # INSTEAD OF THE CURRENT 0-5, 5-10, 10-15 (CAN'T DO THIS BEFORE DOCUMENT COUNT)
        # 3: APPLY DOCUMENT VECTOR
        # STEPS 1 AND 3 SHOULD BE EASYISH
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

    def transform_subblock(self, X_seq):
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
            block_np = block_df.to_numpy()
            all_blocks_count.append(block_np)

        # finally stack the blocks
        X = np.stack(all_blocks_count)

        # CODE BELOW HERE NOT MODIFIED
        # applies tf-idf if parameter
        num_instance, _ = X.shape
        if self.term_weighting == "tf-idf":
            idf_matrix = X * np.tile(self.idf_vec, (num_instance, 1))
            X = idf_matrix

        X_new = X
        print("Test data shape: {}-by-{}\n".format(X_new.shape[0], X_new.shape[1]))
        return X_new


if __name__ == "__main__":

    start = time.time()

    train = pd.read_csv("./train_subset.csv")  # for testing
    # train = pd.read_csv("./HDFS_train.log_structured.csv")
    lab = pd.read_csv("./anomaly_label.csv")
    print("data loaded")

    # Convert to blockId and EventSequence dataframe
    re_pat = r"(blk_-?\d+)"
    col_names = ["BlockId", "EventSequence"]
    events_df = collect_event_ids(train, re_pat, col_names).merge(lab, on="BlockId")

    # Convert label column to binary
    events_df["Label"] = events_df["Label"].apply(lambda x: 1 if x == "Anomaly" else 0)

    # select only events
    events = events_df["EventSequence"].values

    # init feature extractor
    fe = FeatureExtractor()

    # fit and transform x_train
    print("fitting and transforming x train")
    x_train = fe.fit_transform(events, term_weighting="tf-idf")

    # transform x_test
    print("transforming x test")
    x_fake_test = fe.transform(events)

    # x_train and x_fake_test should be equal
    # x_train was fit and transformed
    # x_fake_test was transformed on the fit parameters
    # simple testing, to be removed
    print("are the two results equal? ", np.array_equal(x_train, x_fake_test))

    print("time taken :", time.time() - start)
