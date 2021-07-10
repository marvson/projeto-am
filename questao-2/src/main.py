import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax
from numpy.lib.function_base import average
import pandas as pd
import os.path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from knn import knnbc
import data_analysis
import train_knn
import metrics

# IMPORT DATASET WITH PANDAS
DF_PATH = os.path.dirname(__file__) + "/../data/yeast_csv.csv"
df = pd.read_csv(DF_PATH, encoding="utf-8")

X = np.array(df.iloc[:, :-1])
# TURN LABELS INTO NUMERBS STARTING FROM 0
df.class_protein_localization = pd.factorize(df.class_protein_localization)[0]
y = np.array(df.class_protein_localization)


p = 0.8 # fracao de elementos no conjunto de treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = p, random_state = 42)

# DATA EXPLORATORY ANALYSIS AND PREPROCESSING
#data_analysis.run(df)

# HYPERPARAMETERS TUNNING
#n_neighbors = train_knn.run(df)    # results n=9
n_neighbors=9

# CLASSIFYING:
clf1 = GaussianNB()
clf2 = knnbc(n_neighbors)
# clf3 = JANELA DE PARZEN
clf4 = LogisticRegression(solver='sag', max_iter=100, random_state=42,
                        multi_class='ovr')

y_pred1 = []
y_pred2 = []
y_pred4 = []
for i in range (4):
    y_pred1.append(clf1.fit(X_train,y_train).predict(X_test))


# COMPUTE METRICS
