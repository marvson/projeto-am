"""
==========================================================
  Class probabilities calculated by the VotingClassifier
==========================================================

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
import pandas as pd
import os.path

from sklearn.model_selection import cross_val_score

from knn import knnbc

# NUMBER OF NEIGHBORS TO USE
n_neighbors = 12

# IMPORT YEAST DATASET WITH PANDAS
DF_PATH = os.path.dirname(__file__) + "/../data/yeast_csv.csv"
df = pd.read_csv(DF_PATH, encoding="utf-8")

# COMPUTE SIZE
m = len(df)  # nr of samples=1484
n = len(df.columns) - 1  # nr of features=8

# FEATURES MATRIX
X = np.array(df.iloc[:, :-1])

# TURN LABELS INTO NUMERBS STARTING FROM 0
df.class_protein_localization = pd.factorize(df.class_protein_localization)[0]
y = np.array(df.class_protein_localization)

# TRAINING DATA
xTrain = X[200:, :n]
yTrain = y[200:]

# TESTING DATA
xTest = X[:199, :n]
yTest = y[:199]

# clf1 = NAIVE BAYES
clf2 = knnbc(n_neighbors)
# clf3 = JANELA DE PARZEN
clf4 = LogisticRegression(solver='sag', max_iter=100, random_state=42,
                        multi_class='ovr')

eclf = VotingClassifier(estimators=[('rf', clf2), ('gnb', clf4)],
                        voting='soft',
                        weights=[5, 1])

scores = cross_val_score(clf2, X, y, cv=5)
print("5-fold scores", scores)

# predict class probabilities for all classifiers
probas = [c.fit(xTrain, yTrain).predict_proba(xTest) for c in (clf2, clf4)]
Z = eclf.fit(xTrain,yTrain).predict(xTest)

# Measure mean accuracy fo test data
print("Testing score : %.3f " % (eclf.score(xTest, yTest)))
#print(Z)
