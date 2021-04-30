# Model Folder

Contains the notebook with the CNN log anomaly detection model. A brief description of the model is provided below.

## Model Description

Input data are in the form of the feature matrices described in the [process](../process) folder.

The model model consists of the following architecture:
- Two convolutional layers (16 and 32 filters with 2x2 kernels) with max pooling (2x2 kernels)
- Two MLP hidden layers (120 and 84 nodes)
- An output layer with two nodes representing normal and anomalous labels.
- The convolutional and MLP layers use ReLU activation and the output layer uses softmax.

## Model Results

The model was evaluated using the HDFS data set. The HDFS data were divided into a 80/20 train/test split. The model was trained using the labelled HDFS data and evaluated using the test data.

The model results are provided in the following tables. The metrics indicate that log anomaly detection process is performing extremely well using the HDFS log dataset.

**Training Classification**

|  | True Normal | True Anomalous |
| --- | ---: | ---: |
| **Model Normal** | 305,731 | 22 |
| **Model Anomalous** | 46 | 9,808 |

<br>

**Testing Classification**

|  | True Normal | True Anomalous |
| --- | ---: | ---: |
| **Model Normal** | 118,553 | 5 |
| **Model Anomalous** | 1 | 1978 |

<br>

**Model Performance Metrics**

|  | Precision (%) | Recall (%) | F-Score (%) |
| --- | ---: | ---: | ---: |
| **Training** | 99.5 | 99.8 | 99.7
| **Testing** | 99.9 | 99.7 | 99.8

<br>

## Running the Model

Note that `anomaly_nn.ipynb` can either be run from [colab](https://colab.research.google.com/) or locally.

To Run the model on colab the data need to be in `"./drive/MyDrive/log_data/"`

And to run the model locally the data needs to be in `../process/project_processed_data/`
