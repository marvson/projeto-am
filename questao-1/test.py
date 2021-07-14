import numpy as np
import os.path
import logging
import pandas as pd


from fcm_dfcv import FCM

def example():
    X = np.array([[1, 1], [1, 2], [2, 2], [9, 10], [10, 10], [10, 9], [9, 9], [20,20]])

    fcm = FCM(n_clusters=3, max_iter=10, m=2)
    fcm.set_logger(tostdout=True, level=logging.DEBUG)
    fcm.fit(X, [0, 0, 0, 1, 1, 1, 1, 2])

def iris_example():
    # IMPORT DATASET
    DF_PATH = os.path.dirname(__file__) + "/iris.csv"
    df = pd.read_csv(DF_PATH, encoding="utf-8")

    # FEATURES MATRIX
    X = np.array(df.iloc[:,:-1])

    # TURN LABELS INTO NUMERBS STARTING FROM 0
    df.variety = pd.factorize(df.variety)[0]
    #df.class_protein_localization = pd.factorize(df.class_protein_localization)[0]
    y = np.array(df.iloc[:,-1:])

    fcm = FCM(n_clusters=3, max_iter=15, m=2)
    fcm.set_logger(tostdout=True, level=logging.DEBUG)
    fcm.fit(X, y)

def yeast_example():
    # IMPORT DATASET
    DF_PATH = os.path.dirname(__file__) + "/../questao-2/data/yeast_csv.csv"
    df = pd.read_csv(DF_PATH, encoding="utf-8")

    # FEATURES MATRIX
    X = np.array(df.iloc[:,:-1])

    # TURN LABELS INTO NUMERBS STARTING FROM 0
    df.class_protein_localization = pd.factorize(df.class_protein_localization)[0]
    y = np.array(df.iloc[:,-1:])

    fcm = FCM(n_clusters=10, max_iter=100, m=2)
    fcm.set_logger(tostdout=True, level=logging.DEBUG)
    fcm.fit(X, y)

yeast_example()