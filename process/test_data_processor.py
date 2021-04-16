from data_processor import FeatureExtractor
import numpy as np


def create_test_data():
    """
    creates sample data for testing
    """
    test1 = ["E1", "E2", "E1"]
    test2 = ["E3", "E2", "E5"]
    return np.array([test1, test2])


def test_output_dimensions_after_tf_idf():
    """
    applying tf-idf has numpy reshapes
    this tests that the output dimensions are correct
    """
    test_data = create_test_data()
    fe = FeatureExtractor()
    expected_shape = (2, 20, 4)

    subblocks = fe.fit_transform_subblocks(test_data, term_weighting="tf-idf")

    assert subblocks.shape == expected_shape


def test_fit_and_transform_and_transform():
    """
    simple test that tests the output the fit_transform_subblocks
    is equal to the output from transform_subblocks
    """
    test_data = create_test_data()
    fe = FeatureExtractor()

    subblocks_ft = fe.fit_transform_subblocks(test_data, term_weighting="tf-idf")
    subblocks_t = fe.transform_subblock(test_data)

    assert np.array_equal(subblocks_ft, subblocks_t)


def test_fit_and_transform_with_rolling_dims():
    """
    tests output dimensions of fit_transform_subblocks
    with rolling = True
    """
    test_data = create_test_data()
    fe = FeatureExtractor()
    expected_dims = (2, 19, 4)

    subblocks_ft = fe.fit_transform_subblocks(
        test_data, term_weighting="tf-idf", rolling=True
    )

    assert subblocks_ft.shape == expected_dims


def test_fit_and_transform_with_rolling():
    """
    tests with rolling = True
    """
    test_data = create_test_data()
    fe = FeatureExtractor()

    subblocks_ft = fe.fit_transform_subblocks(
        test_data, term_weighting="tf-idf", rolling=True
    )
    subblocks_t = fe.transform_subblock(test_data)

    assert np.array_equal(subblocks_ft, subblocks_t)
