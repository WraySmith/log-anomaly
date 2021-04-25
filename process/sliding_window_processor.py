"""
loads and preprocesses the structured log data for anomaly prediction
"""
import numpy as np
import pandas as pd
import re
from collections import OrderedDict
from collections import Counter
from PIL import Image


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


def windower(sequence, window_size):
    """
    creates an array of arrays of windows
    output array is of length: len(sequence) - window_size + 1
    """
    return np.lib.stride_tricks.sliding_window_view(sequence, window_size)


def sequence_padder(sequence, required_length):
    """
    right pads events sequence until max sequence length long
    """
    if len(sequence) > required_length:
        return sequence
    return np.pad(
        sequence,
        (0, required_length - len(sequence)),
        mode="constant",
        constant_values=(0),
    )


def resize_time_image(time_image, size):
    """
    compresses time images that had more sequences then the set max sequence length
    """
    width = size[1]
    height = size[0]
    return np.array(Image.fromarray(time_image).resize((width, height)))


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
        self.max_seq_length = None
        self.window_size = None
        self.num_rows = None

    def fit_transform(
        self, X_seq, term_weighting=None, length_percentile=90, window_size=16
    ):
        """
        Fit and transform the training set
        X_Seq: ndarray,  log sequences matrix
        term_weighting: None or `tf-idf`
        length_percentile: int, set the max length of the event sequences
        window_size: int, size of subsetting
        """
        self.term_weighting = term_weighting
        self.window_size = window_size

        # get unique events
        self.events = set(np.concatenate(X_seq).ravel().flatten())

        # get lengths of event sequences
        length_list = np.array(list(map(len, X_seq)))
        self.max_seq_length = int(np.percentile(length_list, length_percentile))

        self.num_rows = self.max_seq_length - self.window_size + 1

        print("final shape will be ", self.num_rows, len(self.events))

        # loop over each sequence to create the time image
        time_images = []
        for block in X_seq:
            padded_block = sequence_padder(block, self.max_seq_length)
            time_image = windower(padded_block, self.window_size)
            time_image_counts = []
            for time_row in time_image:
                row_count = Counter(time_row)
                time_image_counts.append(row_count)

            time_image_df = pd.DataFrame(time_image_counts, columns=self.events)
            time_image_df = time_image_df.reindex(sorted(time_image_df.columns), axis=1)
            time_image_df = time_image_df.fillna(0)
            time_image_np = time_image_df.to_numpy()

            # resize if too large
            if len(time_image_np) > self.num_rows:
                time_image_np = resize_time_image(
                    time_image_np, (self.num_rows, len(self.events)),
                )

            time_images.append(time_image_np)

        # stack all the blocks
        X = np.stack(time_images)

        if self.term_weighting == "tf-idf":

            # set up sizing
            dim1, dim2, dim3 = X.shape
            X = X.reshape(-1, dim3)

            # apply tf-idf
            df_vec = np.sum(X > 0, axis=0)
            self.idf_vec = np.log(dim1 / (df_vec + 1e-8))
            idf_tile = np.tile(self.idf_vec, (dim1 * dim2, 1))
            idf_matrix = X * idf_tile
            X = idf_matrix

            # reshape to original dimensions
            X = X.reshape(dim1, dim2, dim3)

        X_new = X
        print("train data shape: ", X_new.shape)
        return X_new

    def transform(self, X_seq):
        """
        transforms x test
        X_seq : log sequence data
        """

        # loop over each sequence to create the time image
        time_images = []
        for block in X_seq:
            padded_block = sequence_padder(block, self.max_seq_length)
            time_image = windower(padded_block, self.window_size)
            time_image_counts = []
            for time_row in time_image:
                row_count = Counter(time_row)
                time_image_counts.append(row_count)

            time_image_df = pd.DataFrame(time_image_counts, columns=self.events)
            time_image_df = time_image_df.reindex(sorted(time_image_df.columns), axis=1)
            time_image_df = time_image_df.fillna(0)
            time_image_np = time_image_df.to_numpy()

            # resize if too large
            if len(time_image_np) > self.num_rows:
                time_image_np = resize_time_image(
                    time_image_np, (self.num_rows, len(self.events)),
                )

            time_images.append(time_image_np)

        # stack all the blocks
        X = np.stack(time_images)

        if self.term_weighting == "tf-idf":

            # set up sizing
            dim1, dim2, dim3 = X.shape
            X = X.reshape(-1, dim3)

            # apply tf-idf
            idf_tile = np.tile(self.idf_vec, (dim1 * dim2, 1))
            idf_matrix = X * idf_tile
            X = idf_matrix

            # reshape to original dimensions
            X = X.reshape(dim1, dim2, dim3)

        X_new = X
        print("test data shape: ", X_new.shape)
        return X_new


if __name__ == "__main__":

    test_data = "../../project_processed_data/HDFS_100k.log_structured.csv"

    df = pd.read_csv(test_data)

    re_pat = r"(blk_-?\d+)"
    col_names = ["BlockId", "EventSequence"]
    events_df = collect_event_ids(df, re_pat, col_names)

    test_df = events_df.head(100)

    print(test_df.head())

    print(test_df.shape)

    print(test_df["EventSequence"].values)
    lenghts = np.array(list(map(len, test_df["EventSequence"].values)))
    print(lenghts)

    print(max(lenghts))

    test_df.to_csv("../../project_processed_data/test_frame.csv")
