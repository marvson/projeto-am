import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax
from numpy.lib.function_base import average
import pandas as pd
import os.path

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from knn import knnbc
import metrics

import data_analysis

# IMPORT DATASET WITH PANDAS
DF_PATH = os.path.dirname(__file__) + "/../data/yeast_csv.csv"
df = pd.read_csv(DF_PATH, encoding="utf-8")

# 
#df = data_analysis.run(df)

X = np.array(df.iloc[:, :-1])
df.class_protein_localization = pd.factorize(df.class_protein_localization)[0]
y = np.array(df.class_protein_localization)

scores = []
k_vector = range(2,50,1)
for k in k_vector:
    clf2=knnbc(k)
    scores.append(average(cross_val_score(clf2, X, y, cv=5)))
    print(f"Average score for {k} neighbors using 5-fold cross validation: {scores[k-min(k_vector)]}")
scores = np.array(scores)
x=k_vector
y=scores
plt.plot(x,y) 
plt.show()
print(f"Best k value is: {x[argmax(y)]}")
