import numpy as np
import math
import random
import logging

from numpy.core.fromnumeric import argmax, transpose
from numpy.lib.function_base import diff
from sklearn.base import BaseEstimator
from metrics import accuracy_score


SMALL_VALUE = 0.00001
class FCM(BaseEstimator):
    def __init__(self, n_clusters=2, m=2, max_iter=10):
        self.n_clusters = n_clusters    # Number of clusters
        self.G = None                   # cluster centers
        self.M = None                   # M Matrix for each cluster
        self.u = None                   # Membership matrix
        self.J = 0                      # Objective function
        self.m = m                      # the fuzziness, m=1 is hard not fuzzy  
        self.max_iter = max_iter

    def init_membership_random(self, X):
        num_of_points, num_of_features = np.shape(X)
        u = np.zeros((self.n_clusters, num_of_points))
        for k in range(num_of_points):
            row_sum = 0.0
            for i in range(self.n_clusters):
                if i == self.n_clusters-1:  # last iteration
                    u[i][k] = 1.0 - row_sum
                else:
                    rand_clus = random.randint(0, self.n_clusters-1)
                    rand_num = random.random()
                    rand_num = round(rand_num, 2)
                    if rand_num + row_sum <= 1.0:  # to prevent membership sum for a point to be larger than 1.0
                        u[i][k] = rand_num
                        row_sum += u[i][k]
        return u
        
    def update_prototypes(self, X):
        num_of_points, num_of_features = np.shape(X)
        c = self.n_clusters
        g = np.zeros((c,num_of_features))
        for i in range(c):
            num = np.zeros(num_of_features)
            den = 0.0
            for k in range(num_of_points):
                num += (self.u[i][k] ** self.m)*X[k]
                den += (self.u[i][k] ** self.m)
            if den==0:
                den=SMALL_VALUE            
            g[i]=num/den
        return g
    
    def compute_single_lambda(self, X, cluster_index, lambda_index):
        n, p = np.shape(X)
        i = cluster_index
        j = lambda_index
        g = self.G

        num=1
        den=0
        for k in range(n):
            den += (self.u[i][k]**self.m)*(X[k][j]-g[i][j])**2
        for h in range(p):
            sum=0
            for k in range(n):
                sum += (self.u[i][k]**self.m)*(X[k][h]-g[i][h])**2
            num = num*sum
        num = num**(1/p)
        if den==0:
            den=SMALL_VALUE
        _lambda = num/den
        return _lambda
    
    def update_M(self, X):
        n, p = np.shape(X)
        c = self.n_clusters
        g = self.G

        M = np.zeros((c,p))
        for i in range(c): 
            for j in range(p):
                M[i][j] = self.compute_single_lambda(X,i,j)
        return M
    
    def compute_single_u(self,X,cluster_index,datapoint_index):
        n, p = np.shape(X)
        i, k = cluster_index, datapoint_index
        c = self.n_clusters
        g = self.G
        M = self.M
        inv_sum = 0
        for h in range(c):
            # NUMERADOR
            v1 = X[k] - g[i]
            num = np.inner((v1*M[i]),v1)
            # DENOMINADOR
            v2 = X[k] - g[h]
            den = np.inner((v2*M[h]),v2)
            if den==0:
                den=SMALL_VALUE
            inv_sum += (num/den)**(1/(self.m-1))
        if inv_sum==0:
            sum = 1.0 - SMALL_VALUE
        else:
            sum = 1/inv_sum
        return sum

    def update_u(self, X):
        n, p = np.shape(X)
        c = self.n_clusters
        u = np.zeros((c,n))
        for i in range(c):
            for k in range(n):
                u[i][k] = self.compute_single_u(X,i,k)
        return u
    
    def update_J(self, X):
        n, p = np.shape(X)
        c = self.n_clusters
        g = self.G
        M = self.M
        J = 0
        for k in range(n):
            for i in range(c):
                v1 = X[k] - g[i]
                J += (self.u[i][k]**self.m)*np.inner((v1*M[i]),v1)
        return J

    def fit(self, X, y=None):
        
        self.u = self.init_membership_random(X)
        J_old = 0

        for i in range(self.max_iter):
            self.G = self.update_prototypes(X)
            self.M = self.update_M(X)
            self.u = self.update_u(X)
            self.J = self.update_J(X)
            #print("J updated:\n",self.J)

            if abs(self.J-J_old) < SMALL_VALUE:
                break
            else:
                J_old = self.J
        return self

    def predict_proba(self, X):
        u = self.update_u(X)
        return u
    
    def predict(self, X):
        u = self.update_u(X)
        predict_u = argmax(u, axis=0)
        return predict_u
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))