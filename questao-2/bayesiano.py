import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score

def bayesiano(classes,x_train,y_train,x_test,y_test):
    
    
    ####  Realiza a classificacao ####
    # Matriz que armazena as probabilidades para cada classe
    P = pd.DataFrame(data=np.zeros((x_train.shape[0], len(classes))), columns = classes) 
    
    Pc = np.zeros(len(classes)) # Armaze a fracao de elementos em cada classe
    for i in np.arange(0, len(classes)): # Para cada classe
        elements = tuple(np.where(y_train == classes[i])) # elmentos na classe i
        Pc[i] = len(elements)/len(y_train) # Probabilidade pertencer a classe i
        Z = x_train[elements,:][0] # Elementos no conjunto de treinamento
        m = np.mean(Z, axis = 0) # Vetor media
        cv = np.cov(np.transpose(Z)) # Matriz de covariancia
        for j in np.arange(0,x_test.shape[0]): # para cada observacao no conjunto de teste
            x = x_test[j,:]
            # calcula a probabilidade pertencer a cada classe
            pj = multivariate_normal.pdf(x, mean=m, cov=cv, allow_singular=True)
            P[classes[i]][j] = pj*Pc[i]
            
    y_pred = [] # Vetor com as classes preditas
    for i in np.arange(0, x_test.shape[0]):
        c = np.argmax(np.array(P.iloc[[i]]))
        y_pred.append(classes[c])
    #y_pred = np.array(y_pred, dtype=str)
    # calcula a acuracia
    #score = accuracy_score(y_pred, y_test)
    return y_test,y_pred
