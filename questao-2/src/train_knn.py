import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax
from numpy.lib.function_base import average
import pandas as pd
import os.path

from sklearn.model_selection import train_test_split
from knn import knnbc
import metrics

def run(df, num_folds=5, num_steps=50):
    X = np.array(df.iloc[:, :-1])
    df.class_protein_localization = pd.factorize(df.class_protein_localization)[0]
    y = np.array(df.class_protein_localization)

    p = 0.8 # fracao de elementos no conjunto de treinamento
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = p, random_state = 42)

    scores = []
    k_vector = range(2,num_steps,1)
    for k in k_vector:
        clf2=knnbc(k)
        clf2.fit(X_train, y_train)
        scores.append(clf2.score(X_test, y_test))
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