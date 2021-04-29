# Log Anomaly Detection

The log anomaly detection project uses a CNN model to detect anomalous log data. The project was completed as part of the Master of Data Science (MDS) program at the University of British Columbia (UBC).

The log anomaly detector uses the following steps:

- **Parse**: Parsing unstructured log data into a structured format consisting of log event template and log variables.
- **Feature Extraction**: TF-IDF on event counts and sliding windows to generate feature matrices.
- **Log Anomaly Detection Model**: CNN model using the feature matrices as inputs and trained using labelled log data.

A brief overview of the project components is provided below.

## Data

HDFS data was used in this project to test the log anomaly detector. The data is provided by the [Loghub collection](https://github.com/logpai/loghub):
- Shilin He, Jieming Zhu, Pinjia He, Michael R. Lyu. [Loghub: A Large Collection of System Log Datasets towards Automated Log Analytics](https://arxiv.org/abs/2008.06448). *Arxiv*, 2020.

Information on the HDFS data can be found [here](https://github.com/logpai/loghub/tree/master/HDFS).

## Parse

This project uses the Drain log parser available through the [Logparser toolkit](https://github.com/logpai/logparser). The Logparser toolkit provides multiple automated log parsing methods to create structured logs (also referred to as message template extraction). 

A description of Drain is provided at the following link:

- [Drain: An Online Log Parsing Approach with Fixed Depth Tree](https://jiemingzhu.github.io/pub/pjhe_icws2017.pdf), by Pinjia He, Jieming Zhu, Zibin Zheng, and Michael R. Lyu.


The raw unstructured HDFS log data is parsed using Drain to generate structured data in the form of log event templates and log variables. The log variables are used to identify groups of log data identified in this case by HDFS block ids. Log messages with the same block id are grouped together.

## Feature Extraction

 THe 