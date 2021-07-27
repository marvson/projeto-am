import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
import pandas as pd
import math
import os.path


def logistic(classes,x_train,y_train,x_test,y_test):

    clf = LogisticRegression(solver='newton-cg', max_iter=100,
                            multi_class='ovr').fit(x_train, y_train)
    
    y_pred = clf.predict(x_test)
    return y_test,y_pred