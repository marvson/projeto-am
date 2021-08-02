import numpy as np
import os.path
import pandas as pd
from collections import Counter
from sklearn import metrics
from sklearn.metrics import cluster

from fcm_dfcv import FCM

def import_yeast():
    # IMPORT DATASET
    DF_PATH = os.path.dirname(__file__) + "/../questao-2/data/yeast_csv.csv"
    df = pd.read_csv(DF_PATH, encoding="utf-8")

    # FEATURES MATRIX
    X = np.array(df.iloc[:,:-1])

    # TURN LABELS INTO NUMERBS STARTING FROM 0
    df.class_protein_localization = pd.factorize(df.class_protein_localization)[0]
    y = np.array(df.iloc[:,-1:])
    return X, y

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
        print(f"finished partition #{i+1}")
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
    print(f"PC: {PC}")
    MPC = 1-(c/(c-1))*(1-PC)
    return MPC
    
def compute_CE(u):
    c, n = np.shape(u)
    CE = 0
    for i in range(c):
        for k in range(n):
            log_u = np.log(u[i][k])
            CE -= u[i][k]*log_u/n
    return CE

def get_crisp(u):
    c, n = np.shape(u)
    crisp_u = np.zeros_like(u)
    print(Counter(np.argmax(u,axis=0)))
    for k in range(n):
        #print(u[:,k])
        predict_u = np.argmax(u[:,k])
        crisp_u[predict_u][k] = 1
    return crisp_u

def run(X,y):
    m = [1.1, 1.6, 2.0]
    for i in m:
        G, U, J, Js = get_partition(X, y, n_clusters=10, max_iter=150, m=i, n_iter=100)
        MPC, CE = compute_MPC(U), compute_CE(U)
        crisp_u = get_crisp(U)
        contingency = cluster.contingency_matrix(y, crisp_u)
        RS = cluster.adjusted_rand_score(y, crisp_u)
        F1 = metrics.f1_score(y,crisp_u)
        print(f"J: {J}, MPC: {MPC}, CE: {CE}")
        print(f"RS: {RS}, F-Score: {F1}")
        print(G)

X, y = import_yeast()
run(X,y)


  
