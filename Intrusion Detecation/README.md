🛡️ Intrusion Detection System using Machine Learning
📌 Project Overview

This project implements a Network Intrusion Detection System (NIDS) using supervised machine learning techniques. The goal is to classify whether a network connection is Normal or an Anomaly (Intrusion) based on network traffic features.

We used the KDD Cup / NSL-KDD dataset (or equivalent intrusion detection dataset) and applied preprocessing, feature engineering, model training, and evaluation.

📂 Dataset

The dataset contains network traffic records with features such as:

protocol_type – Type of protocol (TCP, UDP, ICMP)

service – Network service (HTTP, FTP, etc.)

flag – Connection status flag

src_bytes, dst_bytes – Number of bytes sent/received

count – Number of connections to the same host

same_srv_rate, diff_srv_rate – Service request rates

dst_host_srv_count, dst_host_same_srv_rate – Host-based traffic features

Target labels:

normal → No intrusion

anomaly → Intrusion detected

⚙️ Project Workflow
1️⃣ Data Preprocessing

Handled categorical features (protocol_type, service, flag) using encoding

Normalized numerical features with StandardScaler

Dropped irrelevant or redundant features (num_outbound_cmds)

2️⃣ Feature Selection

Selected the 10 most relevant features using feature importance

3️⃣ Model Training

Trained multiple models:

Decision Tree Classifier

Random Forest Classifier

Support Vector Machine (SVM)

Logistic Regression

K-Nearest Neighbors (KNN)

Saved the best model and the scaler using Pickle for inference.

4️⃣ Evaluation Metrics

Used the following metrics:

Accuracy

Precision, Recall, F1-Score

Confusion Matrix

📊 Results
Model	Accuracy
Decision Tree	~0.96
Random Forest	~0.98
Logistic Regression	~0.92
KNN	~0.91
SVM	~0.95

(Exact results may vary depending on dataset and parameters)
