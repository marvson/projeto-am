from math import inf
import numpy as np
import os.path
import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from fcm_dfcv import FCM

def example():
    X = np.array([[1, 1], [1, 2], [2, 2], [9, 10], [10, 10], [10, 9], [9, 9], [20,20]])

    fcm = FCM(n_clusters=3, max_iter=10, m=2)
    fcm.fit(X, [0, 0, 0, 1, 1, 1, 1, 2])

def iris_example():
    # IMPORT DATASET
    DF_PATH = os.path.dirname(__file__) + "/iris.csv"
    df = pd.read_csv(DF_PATH, encoding="utf-8")

    # FEATURES MATRIX
    X = np.array(df.iloc[:,:-1])

    # TURN LABELS INTO NUMERBS STARTING FROM 0
    df.variety = pd.factorize(df.variety)[0]
    y = np.array(df.iloc[:,-1:])

    p = 0.8 # fracao de elementos no conjunto de treinamento
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = p, random_state = 42)

    fcm = FCM(n_clusters=3, max_iter=150, m=2)

    print("score: ", fcm.fit(X_train, y_train).score(X_test,y_test))
    return fcm.J

def yeast_example():
    # IMPORT DATASET
    DF_PATH = os.path.dirname(__file__) + "/../questao-2/data/yeast_csv.csv"
    df = pd.read_csv(DF_PATH, encoding="utf-8")

    # FEATURES MATRIX
    X = np.array(df.iloc[:,:-1])

    # TURN LABELS INTO NUMERBS STARTING FROM 0
    df.class_protein_localization = pd.factorize(df.class_protein_localization)[0]
    y = np.array(df.iloc[:,-1:])

    p = 0.8 # fracao de elementos no conjunto de treinamento
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = p, random_state = 42)

    fcm = FCM(n_clusters=10, max_iter=150, m=2)
    print("score: ", fcm.fit(X_train, y_train).score(X_test,y_test))
    return fcm.J

J = []
<<<<<<< HEAD
for i in range(10):
=======
for i in range(30):
>>>>>>> development
    J.append(yeast_example())
    print(f"partition #{i+1} objective function value: {J[i]}")
print(min(J))