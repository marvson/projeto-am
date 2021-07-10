import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax
from numpy.lib.function_base import average
import pandas as pd
import os.path

from sklearn.model_selection import cross_val_score
from knn import knnbc
import metrics

def run(df, num_folds=5, num_steps=50):
    X = np.array(df.iloc[:, :-1])
    df.class_protein_localization = pd.factorize(df.class_protein_localization)[0]
    y = np.array(df.class_protein_localization)

    scores = []
    k_vector = range(2,num_steps,1)
    for k in k_vector:
        clf2=knnbc(k)
        scores.append(average(cross_val_score(clf2, X, y, cv=num_folds)))
        print(f"Average score for {k} neighbors using 5-fold cross validation: {scores[k-min(k_vector)]}")
    scores = np.array(scores)
    # PLOT K:
    x=k_vector
    y=scores
    k_max=x[argmax(y)]
    plt.plot(x,y) 
    plt.show()
    print(f"Best k value is: {k_max}")

    return k_max
