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
    """
        This algorithm is from the paper
        "FCM: The fuzzy c-means clustering algorithm" by James Bezdek
        Here we will use the Euclidean distance

        Pseudo code:
        1) Fix c, m, A
        c: n_clusters
        m: 2 by default
        A: we are using Euclidean distance, so we don't need it actually
        2) compute the means (cluster centers)
        3) update the membership matrix
        4) compare the new membership with the old one, is difference is less than a threshold, stop. otherwise
            return to step 2)
    """

    def __init__(self, n_clusters=2, m=2, max_iter=10):
        self.n_clusters = n_clusters
        self.cluster_centers_ = None
        self.cluster_M = [] # Matrizes M
        self.u = None  # The membership matrix
        self.J = 0
        self.m = m  # the fuzziness, m=1 is hard not fuzzy  
        self.max_iter = max_iter
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.NullHandler())
        # logger.addHandler(logging.StreamHandler())

    def get_logger(self):
        return self.logger

    def set_logger(self, tostdout=False, logfilename=None, level=logging.WARNING):
        if tostdout:
            self.logger.addHandler(logging.StreamHandler())
        if logfilename:
            self.logger.addHandler(logging.FileHandler(logfilename))
        if level:
            self.logger.setLevel(level)

    def init_membership(self, num_of_points):
        self.init_membership_random(num_of_points)

    def init_membership_random(self, num_of_points):
        """
        :param num_of_points:
        :return: nothing

        """
        self.u = np.zeros((self.n_clusters, num_of_points))
        for k in range(num_of_points):
            row_sum = 0.0
            for i in range(self.n_clusters):
                if i == self.n_clusters-1:  # last iteration
                    self.u[i][k] = 1.0 - row_sum
                else:
                    rand_clus = random.randint(0, self.n_clusters-1)
                    rand_num = random.random()
                    rand_num = round(rand_num, 2)
                    if rand_num + row_sum <= 1.0:  # to prevent membership sum for a point to be larger than 1.0
                        self.u[i][k] = rand_num
                        row_sum += self.u[i][k]
        #print(self.u)
        
    def update_cluster_centers(self, X):
        """
        :param X:
        :return: cluster centers
        g[i] = SUM(u[i][k]^m*x[k])/SUM(u[i][k]^m)
        """
        num_of_points, num_of_features = np.shape(X)
        g = np.zeros((self.n_clusters,num_of_features))
        for i in range(self.n_clusters):
            num = np.zeros(num_of_features)
            den = 0.0
            for k in range(num_of_points):
                num += (self.u[i][k] ** self.m)*X[k]
                den +=  self.u[i][k] ** self.m              
            g[i]=num/den
        self.cluster_centers_ = g
        return g
    
    def compute_objective_function(self, X):
        """
        Compute objective function based on an adaptive quadratic distance 
        defined by a diagonal fuzzy covariance matrix (DFCM)
        :param X: 
        :return: clustering criterion
        """
        num_of_points, num_of_features = np.shape(X)
        g = self.cluster_centers_
        J = 0
        for i in range(self.n_clusters):
            M = self.cluster_M[i]
            for k in range(num_of_points):
                vec = X[k]-g[i]
                vecT = vec[...,None]
                J+=(self.u[i][k] ** self.m)*np.matmul(np.matmul(vec,M),vecT)
        return J
    
    def compute_cluster_lambda(self, X, cluster_index):
        num_of_points, num_of_features = np.shape(X)
        g = self.cluster_centers_
        i = cluster_index
        M = np.zeros((num_of_features,num_of_features))
        for j in range(num_of_features):
            num_prod = 1
            den = 0
            for k in range(num_of_points):
                den += (self.u[i][k] ** self.m)*(X[k][j]-g[i][j])**2
            for h in range(num_of_features):
                num_sum = 0
                for k in range(num_of_points):
                    num_sum += (self.u[i][k] ** self.m)*(X[k][h]-g[i][h])**2
                num_prod = num_prod*num_sum 
            num = num_prod**(1/num_of_features)
            if den == 0:
                _lambda = 1 - SMALL_VALUE
            else: _lambda = num/den
            M[j][j] = _lambda
        return M
    
    def update_cluster_lambdas(self, X):
        M = []
        for i in range(self.n_clusters):
            M.append(self.compute_cluster_lambda(X, i))
        return M

    def compute_membership_single(self, X, cluster_index, datapoint_index):
        """
        :param X:
        :param cluster_index:
        :param datapoint_index:
        :return: compute membership for the given indexes
        """
        num_of_points, num_of_features = np.shape(X)
        i = cluster_index
        k = datapoint_index
        g = self.cluster_centers_
        inv_sum = 0
        for h in range(self.n_clusters):
            # Numerador
            M_num = self.cluster_M[i]
            vec = X[k] - g[i]
            vecT = vec[...,None]
            num = np.matmul(np.matmul(vec,M_num),vecT)
            # Denominador
            M_den = self.cluster_M[h]
            vec = X[k] - g[h]
            vecT = vec[...,None]
            den = np.matmul(np.matmul(vec,M_den),vecT)
            if den == 0:
                den=SMALL_VALUE
            # Inverso do Somatório
            inv_sum += (num/den)**(1/(self.m-1))
        if inv_sum == 0:
            sum = 1.0 - SMALL_VALUE
        else:
            sum = 1/inv_sum        
        #print("sum: ", sum)
        return sum            

    def update_membership(self, X):
        """
        update the membership matrix
        :param X: data points
        :return: nothing

        For performance, the distance can be computed once, before the loop instead of computing it every time
        """
        num_of_points, num_of_features = np.shape(X)
        for i in range(self.n_clusters):
            for k in range(num_of_points):
                self.u[i][k] = self.compute_membership_single(X, i, k)
    
    def compute_data_membership(self, X):
        """
        update the membership matrix
        :param X: data points
        :return: nothing

        For performance, the distance can be computed once, before the loop instead of computing it every time
        """
        num_of_points, num_of_features = np.shape(X)
        u = np.zeros((self.n_clusters,num_of_points))
        for i in range(self.n_clusters):
            for k in range(num_of_points):
                u[i][k] = self.compute_membership_single(X, i, k)
        return u           

    def fit(self, X, y=None):
        """
        :param X:
        :param y: list of clusters or a membership, now only support the hard y list which will generate
        the membership
        :return: self
        """
        X = np.array(X)
        num_of_points, num_of_features = np.shape(X)
        
        self.init_membership_random(num_of_points)
        J_old = 0

        # START FITTING CENTROIDS
        for i in range(self.max_iter):
            # FIRST STAGE: UPDATE PROTOTYPES
            centers = self.update_cluster_centers(X)

            # SECOND STAGE: UPDATE CLUSTERING MATRICES
            self.cluster_M = self.update_cluster_lambdas(X)

            # THIRD STAGE: ALLOCATE MEMBERSHIP DEGREES
            self.update_membership(X)
            #self.logger.info("membership succesfully updated")

            # COMPUTE OBJECTIVE
            self.J = self.compute_objective_function(X)
            #self.logger.info("updated objective function value is: ")
            #self.logger.info(self.J)
            if abs(self.J-J_old) < SMALL_VALUE:
                break
            else:
                J_old = self.J
        #diagonal_prod = [np.product(np.diag(self.cluster_M[i])) for i in range (self.n_clusters)]
        #print(diagonal_prod)
        return self

    def predict_proba(self, X):
        num_of_points, num_of_features = np.shape(X)
        u = np.zeros((self.n_clusters, num_of_points))
        u = self.compute_data_membership(X)
        #predicted_u = self.u.copy()
        if np.any(np.isnan(u)):
            self.logger.debug("predict> has a nan")
            self.logger.debug("u:")
            self.logger.debug(u)
            raise Exception("There is a nan in predict method")
        return u

    def predict(self, X):
        num_of_points, num_of_features = np.shape(X)
        u = np.zeros((self.n_clusters, num_of_points))
        u = self.compute_data_membership(X)
        #predicted_u = self.u.copy()
        if np.any(np.isnan(u)):
            self.logger.debug("predict> has a nan")
            self.logger.debug("u:")
            self.logger.debug(u)
            raise Exception("There is a nan in predict method")
        predict_u = argmax(u, axis=0)
        return predict_u


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