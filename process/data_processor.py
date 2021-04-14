"""
loads and preprocesses the structured log data for anomoly prediction
"""
import numpy as np
import pandas as pd
import time
import re
from collections import OrderedDict
from collections import Counter
from scipy.special import expit


def collect_event_ids(data_frame):
    """
    turns input data_frame into a 2 columned dataframe
    with columns: BlockId, EventSequence
    where EventSequence is a list of the events that happened to the block
    """
    data_dict = OrderedDict()
    for _, row in data_frame.iterrows():
        blk_id_list = re.findall(r"(blk_-?\d+)", row["Content"])
        blk_id_set = set(blk_id_list)
        for blk_id in blk_id_set:
            if not blk_id in data_dict:
                data_dict[blk_id] = []
            data_dict[blk_id].append(row["EventId"])
    data_df = pd.DataFrame(
        list(data_dict.items()), columns=["BlockId", "EventSequence"]
    )
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
        self.normalization = None

    def fit_transform(self, X_seq, term_weighting=None, normalization=None):
        """
        Fit and transform the training set
        X_Seq: ndarray,  log sequences matrix
        term_weighting: None or `tf-idf`
        normalization: None - not implemented yet
        """
        self.term_weighting = term_weighting
        self.normalization = normalization

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

        # normalization if parameter
        if self.normalization == "zero-mean":
            mean_vec = X.mean(axis=0)
            self.mean_vec = mean_vec.reshape(1, num_event)
            X = X - np.tile(self.mean_vec, (num_instance, 1))
        elif self.normalization == "sigmoid":
            X[X != 0] = expit(X[X != 0])  # expit is logistic sigmoid for ndarrays

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

        # applies tf-idf if pararmeter
        num_instance, _ = X.shape
        if self.term_weighting == "tf-idf":
            idf_matrix = X * np.tile(self.idf_vec, (num_instance, 1))
            X = idf_matrix

        # applies normalization if parameter
        if self.normalization == "zero-mean":
            X = X - np.tile(self.mean_vec, (num_instance, 1))
        elif self.normalization == "sigmoid":
            X[X != 0] = expit(X[X != 0])
        X_new = X
        print("Test data shape: {}-by-{}\n".format(X_new.shape[0], X_new.shape[1]))
        return X_new


if __name__ == "__main__":

    start = time.time()

    # train = pd.read_csv("./train_subset.csv") # for testing
    train = pd.read_csv("./HDFS_train.log_structured.csv")
    lab = pd.read_csv("./anomaly_label.csv")
    print("data loaded")

    # Convert to blockId and EventSequence dataframe
    events_df = collect_event_ids(train).merge(lab, on="BlockId")

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
