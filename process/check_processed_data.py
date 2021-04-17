import numpy as np
import pandas as pd


if __name__ == "__main__":

    y_train = pd.read_csv("y_train.csv")
    y_test = pd.read_csv("y_test.csv")

    x_train = np.load("./x_train.npy")
    x_test = np.load("./x_test.npy")

    print("this is the train data")
    print(x_train.shape)
    print(y_train.shape)

    print("this is the test data")
    print(x_test.shape)
    print(y_test.shape)
