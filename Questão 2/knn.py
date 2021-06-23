from datasetAnalysis import X
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import neighbors, datasets
import pandas as pd

# NUMBER OF NEIGHBORS TO USE
n_neighbors = 7

# IMPORT YEAST DATASET WITH PANDAS
DF_PATH = 'yeast_csv.csv'
df= pd.read_csv(DF_PATH, encoding= 'utf-8')

# COMPUTE SIZE
m=len(df) #nr of samples=1484
n=len(df.columns)-1 #nr of features=8

# FEATURES MATRIX
X=np.array(df.iloc[:,:n])

# TURN LABELS INTO NUMERBS STARTING FROM 0
df.class_protein_localization = pd.factorize(df.class_protein_localization)[0]
y=np.array(df.class_protein_localization)

# NUMBER OF LABELS = 10
labels = len(set(y))
print(labels)

# TRAINING DATA
xTrain=X[200:,:n]
yTrain=y[200:]

# TESTING DATA
xTest=X[:199,:n]
yTest=y[:199]

# Create an instance of Neighbours Classifier and fit the data
clf = neighbors.KNeighborsClassifier(n_neighbors)
clf.fit(xTrain, yTrain)

# TEST PREDICTION
Z = clf.predict(xTest)
#Z = clf.predict_proba(xTest)

#print(yTest)
#print(Z)

# COUNTS HOW MANY PREDICTIONS HIT
count = np.sum(yTest == Z)
print(count/200)