from operator import pos
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn import neighbors, datasets
import pandas as pd
import math
import os.path
from sklearn.preprocessing import StandardScaler
import statistics
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

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
        #self.n_labels = len(y)

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
            #for k in classes:
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
            #for k in classes:    
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


# data = pd.read_csv('yeast.data', header=(0))
# data = data.drop(columns=["a1"])
# classes = np.array(pd.unique(data[data.columns[-1]]), dtype=str)
# #data.classe = pd.factorize(data.classe)[0]
# data = data.to_numpy()
# nrow,ncol = data.shape
# y = data[:,-1]
# x = data[:,0:ncol-1]

# # Transforma os dados para terem media igual a zero e variancia igual a 1
# scaler = StandardScaler().fit(x)
# x = scaler.transform(x)

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8)


def knn(classes,x_train,y_train,x_test,y_test):

    # converter as classes para forma numérica
    # y_test=pd.DataFrame(y_test)
    # y_test = pd.factorize(y_test[0])[0]
    # y_train=pd.DataFrame(y_train)
    # y_train = pd.factorize(y_train[0])[0]
    
    # validação do k    
    score_val=[]
    k_list=[]
    
    kfold = KFold(n_splits=5, shuffle=True)
    
    # validação do "k"
    for k in range(2,15):
    #for k in range(2,3):
        score_tot=0
        for train, test in kfold.split(x_train):
            knn=knnbc(k)
            knn.fit(x_train[train],y_train[train])   ########
            score=knn.score(x_train[test],y_train[test])
            score_tot=score_tot+score
        score_val.append(score_tot/5)
        k_list.append(k)
    k_best=k_list[score_val.index(np.max(score_val))]
    #plt.plot (k_list,np.transpose(np.array(score_val)))
    
    # treinamento e teste
    knn=knnbc(k_best)
    knn.fit(x_train,y_train)   ########
    y_pred=knn.predict(x_test)  ########
    
    return y_test, y_pred