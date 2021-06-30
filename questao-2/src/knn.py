from operator import pos
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax
import seaborn as sns
from sklearn import neighbors, datasets
import pandas as pd
import math
import os.path

# NUMBER OF NEIGHBORS TO USE
n_neighbors = 10

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

# NUMBER OF LABELS = 10
labels = len(set(y))
# print(labels, "labels")

# TRAINING DATA
xTrain = X[200:, :n]
yTrain = y[200:]

# TESTING DATA
xTest = X[:199, :n]
yTest = y[:199]

# Compute prior probabilities p(Ck)
unique, prioriProb = np.unique(yTrain, return_counts=True)
pCk = prioriProb/len(yTrain) 

# Compute the likelihoods for each class
clf = neighbors.NearestNeighbors(n_neighbors=n_neighbors)
K = n_neighbors     # Number of neighbors
D = 8               # Number of dimensions
# pXC = []            # Class-conditional densities: P(X|Ck)
# pX = []             # Evidences: P(X)
pCX = []            # Posterior probability: P(Ck|X)
# Since D is odd, volume of unit sphere is:
# C = pi^(D/2)/(D/2)! => C = pi^(4)/2
C = pow(math.pi, D/2)/math.factorial(D/2)
# Make predicitions:
Z = []
for Xi in xTest:
    PXi=[]                                          # Init densities vector for Xi
    evidence=0
    posterior=0
    for k in range(0, labels):                      # Run for all classes
        XCk = []                                    # Init matrix for computing X's from each Class k
        for j in range(0, len(yTrain)):             # Run for whole training dataset
            if k == yTrain[j]:                      # 
                XCk.append(xTrain[j])               # XCk <= new sample if it is labeled as Class k
        clf.fit(XCk)                                # Fits the nearest neighbor classifier based on XCk
        n=n_neighbors                               # n <= number of neighbors to search
        while(True):
            try:                                    # If Xi does not have enough neighbors of class Ck, try with less
                R, index = clf.kneighbors([Xi],n)   # R <= distances from Xi to its k-neighbors
                break                               # Proceed if R is sucessfully computed
            except:
                n=n-1                               # Reduce number of neighbors
        Rk = max(max(R))                            # Rk <= max value between R's
        V = C*Rk                                    # Volume for Rk
        PXi.append(K/(len(XCk)*V))                  # Density for Xi relative to Ck
    evidence=np.inner(PXi,pCk)                      # Compute evidence for Xi
    # pX.append(evidence)                           # pX <= new sample of evidence P(Xi)                
    # pXC.append(PXi)                               # pXC <= new sample of Class Conditional densities
    posterior = PXi*pCk/evidence                    # Compute posterior probabilities for Xi
    pCX.append(posterior)                           # pCX <= new sample of posterior probs P(Ck|Xi)
    Z.append(argmax(posterior))                     # Z <= index of most likely label for Xi                   
# print(Z)

# TEST PREDICTION
# clf = neighbors.KNeighborsClassifier(n_neighbors)
# clf.fit(xTrain, yTrain)
# Z = clf.predict(xTest)
# Z = clf.predict_proba(xTest)


# print(yTest)
# print(Z)

# COUNTS HOW MANY PREDICTIONS HIT
count = np.sum(yTest == Z)
print(count / 200, "percent of the predictions hit")