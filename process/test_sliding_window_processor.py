"""
tests functionality
"""

import sliding_window_processor as dp
import numpy as np
import pandas as pd


def load_test_data():
    "loads test data of 100 rows"
    df = pd.read_csv("./testing_frame.csv", converters={"EventSequence": eval})
    return df


def test_windower():
    """
    creates test sequence and then windows and tests output
    """
    test_sequence = ["E{}".format(str(x)) for x in range(0, 256)]

    window_size = 32
    output = dp.windower(test_sequence, window_size)
    for item in output:
        assert len(item) == window_size
    assert len(output) == (256 - 32 + 1)

    counter = 0
    for returned in output:
        expected = ["E{}".format(str(x)) for x in range(counter, window_size + counter)]
        assert (returned == expected).all()
        counter += 1

    window_size = 16
    output = dp.windower(test_sequence, window_size)
    for item in output:
        assert len(item) == window_size
    assert len(output) == (256 - 16 + 1)

    counter = 0
    for returned in output:
        expected = ["E{}".format(str(x)) for x in range(counter, window_size + counter)]
        assert (returned == expected).all()
        counter += 1


def test_sequence_padder():
    """
    tests function for correct length and correct values
    """
    test_data = np.array([1, 1, 2, 3])
    required_length = 32

    returned = dp.sequence_padder(test_data, required_length)
    assert len(returned) == 32
    expected = np.concatenate((test_data, np.array([0] * 28)))
    assert np.array_equal(returned, expected)

    returned = dp.sequence_padder(test_data, 100)
    assert len(returned) == 100
    expected = np.concatenate((test_data, np.array([0] * 96)))
    assert np.array_equal(returned, expected)

    returned = dp.sequence_padder(test_data, 2)
    assert len(returned) == 4
    expected = np.array([1, 1, 2, 3])
    assert np.array_equal(returned, expected)


def test_fit_transfom():
    """
    tests output of fit_transform
    """
    data = load_test_data()
    events = data["EventSequence"].values
    fe = dp.FeatureExtractor()

    X = fe.fit_transform(events, length_percentile=100, window_size=32)
    assert X.shape == (100, 218, 14)

    X = fe.fit_transform(events, length_percentile=100, window_size=2)
    assert X.shape == (100, 248, 14)

    X = fe.fit_transform(events, length_percentile=100, window_size=7)
    assert X.shape == (100, 249 - 7 + 1, 14)

    test_sequence = [["E1", "E2", "E3"] * 10, ["E1", "E3"] * 5]
    X = fe.fit_transform(test_sequence, length_percentile=50, window_size=2)
    assert X.shape == (2, 20 - 2 + 1, 3)

    X = fe.fit_transform(
        events, length_percentile=100, window_size=32, term_weighting="tf-idf"
    )
    assert X.shape == (100, 218, 14)


def test_transform():
    """
    tests output of transform
    """
    data = load_test_data()
    events = data["EventSequence"].values
    fe = dp.FeatureExtractor()

    # test fit_transform and transform change the same data in the same way
    X = fe.fit_transform(events, length_percentile=100, window_size=32)
    assert X.shape == (100, 218, 14)
    X_t = fe.transform(events)
    assert X_t.shape == (100, 218, 14)
    assert np.array_equal(X, X_t)

    # test that a test set loses events that are not in the train set
    sequence1 = [["E1", "E2", "E3"] * 10, ["E1", "E3", "E4"] * 10]
    X = fe.fit_transform(sequence1, length_percentile=100, window_size=10)
    sequence2 = [["E1", "E2", "E9"] * 10, ["E1", "E3", "E4", "E10"] * 10]
    X_t = fe.transform(sequence2)
    assert X.shape == (2, 30 - 10 + 1, 4)
    assert X_t.shape == (2, 30 - 10 + 1, 4)


def test_resize_time_image():
    """
    test image scaling
    """
    test_1 = np.array(
        [
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )

    test_2 = np.array(
        [
            [2.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [2.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [2.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [2.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    test_1_reshape = dp.resize_time_image(test_1, (10, 3))
    assert test_1_reshape.shape == (10, 3)

    test_2_reshape = dp.resize_time_image(test_2, (10, 3))
    assert test_2_reshape.shape == (10, 3)
    assert np.allclose(test_2, test_2_reshape)
