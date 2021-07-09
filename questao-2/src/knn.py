from operator import pos
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax
from metrics import accuracy_score
import seaborn as sns
from sklearn import neighbors, datasets
import pandas as pd
import math
import os.path

from sklearn.base import BaseEstimator

class knnbc(BaseEstimator):
    """ 
    Custom KNN based Bayesian Classifier

    """
    def __init__(self, n_neighbors=5):
        super(knnbc, self).__init__()           # Base class method
        self._estimator_type = "classifier"
        self.n_neighbors = n_neighbors
        self._fit_X = None
        self._y = None
        self.priori_probs = None
        self.n_features = None
        self.n_samples = None        
        self.n_labels = None
        self.unit_sphere_vol = None

    def fit(self, X, y):
        """
        Method for fitting knn training data

        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]

        y : numpy array, shape [n_samples, ]

        """
        # SAVE KNN DATA
        self._fit_X = X
        self._y = y

        # COMPUTE SIZE
        self.n_samples, self.n_features = X.shape    # num_rows, num_columns

        # NUMBER OF LABELS = 10
        self.n_labels = len(set(y))

        # Compute prior probabilities p(Ck)
        unique, prioriProb = np.unique(y, return_counts=True)
        pCk = prioriProb/len(y)
        self.priori_probs = pCk

        # Compute the likelihoods for each class
        K = self.n_neighbors         # Number of neighbors
        D = self.n_features     # Number of dimensions
        # Since D is odd, volume of unit sphere is:
        # C = pi^(D/2)/(D/2)! => C = pi^(4)/2 
        self.unit_sphere_vol = pow(math.pi, D/2)/math.factorial(D/2)

        return self

    def predict(self, X):
        """
        Method for predicting new data using KNN bBsed Bayesian Classifier

        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]
        
        """
        clf = neighbors.NearestNeighbors(n_neighbors=self.n_neighbors)
        yTrain = self._y
        xTrain = self._fit_X
        C = self.unit_sphere_vol
        K = self.n_neighbors
        pCX = []            # Posterior probability: P(Ck|X)
        Z = []
        for Xi in X:
            PXi=[]                                          # Init densities vector for Xi
            evidence=0
            posterior=0
            for k in range(0, self.n_labels):               # Run for all classes
                XCk = []                                    # Init matrix for computing X's from each Class k
                for j in range(0, len(yTrain)):             # Run for whole training dataset
                    if k == yTrain[j]:                      # 
                        XCk.append(xTrain[j])               # XCk <= new sample if it is labeled as Class k
                clf.fit(XCk)                                # Fits the nearest neighbor classifier based on XCk
                for N in range (K, 0, -1):
                    try:                                    # If Xi does not have enough neighbors of class Ck, try with less
                        R, index = clf.kneighbors([Xi],N)   # R <= distances from Xi to its k-neighbors
                        break                               # Proceed if R is sucessfully computed
                    except:
                        N=N-1                               # Reduce number of neighbors
                Rk = max(max(R))                            # Rk <= max value between R's
                V = C*Rk                                    # Volume for Rk
                PXi.append(K/(len(XCk)*V))                  # Density for Xi relative to Ck
            evidence=np.inner(PXi,self.priori_probs)        # Compute evidence for Xi
            # pX.append(evidence)                           # pX <= new sample of evidence P(Xi)                
            # pXC.append(PXi)                               # pXC <= new sample of Class Conditional densities
            posterior = PXi*self.priori_probs/evidence      # Compute posterior probabilities for Xi
            pCX.append(posterior)                           # pCX <= new sample of posterior probs P(Ck|Xi)
            Z.append(argmax(posterior))                     # Z <= index of most likely label for Xi                   

        return Z


    def predict_proba(self, X):
        """
        Method for predicting new data using KNN bBsed Bayesian Classifier

        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]
        
        """
        clf = neighbors.NearestNeighbors(n_neighbors=self.n_neighbors)
        yTrain = self._y
        xTrain = self._fit_X
        C = self.unit_sphere_vol
        K = self.n_neighbors
        pCX = []            # Posterior probability: P(Ck|X)
        for Xi in X:
            PXi=[]                                          # Init densities vector for Xi
            evidence=0
            posterior=0
            for k in range(0, self.n_labels):               # Run for all classes
                XCk = []                                    # Init matrix for computing X's from each Class k
                for j in range(0, len(yTrain)):             # Run for whole training dataset
                    if k == yTrain[j]:                      # 
                        XCk.append(xTrain[j])               # XCk <= new sample if it is labeled as Class k
                clf.fit(XCk)                                # Fits the nearest neighbor classifier based on XCk
                for N in range (K, 0, -1):
                    try:                                    # If Xi does not have enough neighbors of class Ck, try with less
                        R, index = clf.kneighbors([Xi],N)   # R <= distances from Xi to its k-neighbors
                        break                               # Proceed if R is sucessfully computed
                    except:
                        N=N-1                               # Reduce number of neighbors
                Rk = max(max(R))                            # Rk <= max value between R's
                V = C*Rk                                    # Volume for Rk
                PXi.append(K/(len(XCk)*V))                  # Density for Xi relative to Ck
            evidence=np.inner(PXi,self.priori_probs)        # Compute evidence for Xi
            # pX.append(evidence)                           # pX <= new sample of evidence P(Xi)                
            # pXC.append(PXi)                               # pXC <= new sample of Class Conditional densities
            posterior = PXi*self.priori_probs/evidence      # Compute posterior probabilities for Xi
            pCX.append(posterior)                           # pCX <= new sample of posterior probs P(Ck|Xi)                 

        return pCX

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.
        """
        return accuracy_score(y, self.predict(X))

