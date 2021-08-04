import numpy as np
import os.path
import pandas as pd
from collections import Counter
from sklearn import metrics
from sklearn.metrics import cluster

from fcm_dfcv import FCM

def import_yeast():
    # IMPORT DATASET
    DF_PATH = os.path.dirname(__file__) + "/../data/yeast_csv.csv"
    df = pd.read_csv(DF_PATH, encoding="utf-8")

    # FEATURES MATRIX
    X = np.array(df.iloc[:,:-1])

    # TURN LABELS INTO NUMERBS STARTING FROM 0
    labels = pd.factorize(df.class_protein_localization)[0]
    y = np.array(labels)

    return X, y, labels

def get_partition(X, y, n_clusters=10, max_iter=150, m=2, n_iter=100):
    clf = FCM(n_clusters=n_clusters, max_iter=max_iter, m=m)
    J_old = np.inf
    Js = []
    for i in range(n_iter):
        clf.fit(X,y)
        Js.append(clf.J)
        if clf.J < J_old:
            if i>0: print(f"{clf.J} < {J_old}")
            G, U, J = clf.G, clf.u, clf.J
            J_old = J
        #print(f"finished partition #{i+1}")
    return G, U, J, Js

def compute_PC(u):
    c, n = np.shape(u)
    PC = 0
    for i in range(c):
        for k in range(n):
            PC += (u[i][k]**2)/n
    return PC
    
def compute_MPC(u):
    c, n = np.shape(u)
    PC = compute_PC(u)
    #print(f"PC: {PC}")
    MPC = 1-(c/(c-1))*(1-PC)
    return MPC
    
def compute_CE(u):
    c, n = np.shape(u)
    CE = 0
    for i in range(c):
        for k in range(n):
            log_u = np.log(u[i][k])
            CE -= u[i][k]*(log_u)/n
    return CE

def get_crisp(u):
    c, n = np.shape(u)
    crisp_u = np.zeros_like(u)
    print(Counter(np.argmax(u,axis=0)))
    for k in range(n):
        #print(u[:,k])
        predict_u = np.argmax(u[:,k])
        crisp_u[predict_u][k] = 1
    return crisp_u, np.argmax(u, axis=0)

def accuracy(contingency):
    return sum(np.max(contingency,axis=0))/np.sum(contingency)

def run(X,y,labels):
    m = [1.1, 1.6, 2.0]
    for i in m:
        G, U, J, Js = get_partition(X, y, n_clusters=10, max_iter=150, m=i, n_iter=100)
        MPC, CE = compute_MPC(U), compute_CE(U)
        crisp_u, predict_u = get_crisp(U)
        contingency = cluster.contingency_matrix(labels, predict_u)
        RS = cluster.adjusted_rand_score(labels, predict_u)
        F1 = metrics.f1_score(labels, predict_u, average='weighted', zero_division=0)
        OERC = 1-accuracy(contingency)
        with open(os.path.dirname(__file__) + "/output.txt", "a") as text_file:
            print(f"Para m={i}", file=text_file)
            print(f"Matriz de Prototipos:\n{G}", file=text_file)
            print(f"Matriz de Confusao:\n{contingency}", file=text_file)
            print(f"J: {J}, MPC: {MPC}, CE: {CE}", file=text_file)
            print(f"RS: {RS}, F-Score: {F1}, OERC:{OERC}", file=text_file)
            print(f"{Counter(np.argmax(U,axis=0))}", file=text_file)   

X, y, labels = import_yeast()
run(X,y,labels)

