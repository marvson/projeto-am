"""
====================================================
         One-vs-Rest Logistic Regression
====================================================

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
import pandas as pd
import math
import os.path

# IMPORT YEAST DATASET WITH PANDAS
DF_PATH = os.path.dirname(__file__) + "/../data/yeast_csv.csv"
df = pd.read_csv(DF_PATH, encoding="utf-8")

# COMPUTE SIZE
m = len(df)  # nr of samples=1484
n = len(df.columns) - 1  # nr of features=8

# FEATURES MATRIX
X = np.array(df.iloc[:, :n])

# TURN LABELS INTO NUMERBS STARTING FROM 0
df.class_protein_localization = pd.factorize(df.class_protein_localization)[0]
y = np.array(df.class_protein_localization)

# TRAINING DATA
xTrain = X[200:, :n]
yTrain = y[200:]

# TESTING DATA
xTest = X[:199, :n]
yTest = y[:199]

# make 3-class dataset for classification
# centers = [[-5, 0], [0, 1.5], [5, -1]]
# X, y = make_blobs(n_samples=1000, centers=centers, random_state=40)
# transformation = [[0.4, 0.2], [-0.4, 1.2]]
# X = np.dot(X, transformation)

multi_class = 'ovr'


clf = LogisticRegression(solver='sag', max_iter=100, random_state=42,
                        multi_class=multi_class).fit(xTrain, yTrain)

# print the training scores
print("Training score : %.3f (%s)" % (clf.score(xTrain, yTrain), multi_class))

Z = clf.predict(xTest)

# Measure mean accuracy fo test data
print("Testing score : %.3f (%s)" % (clf.score(xTest, yTest), multi_class))
