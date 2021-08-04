# Classificador Bayesiano Multivariado utilizando kernel gaussiano
# com janela de Parzen

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from sklearn.metrics import accuracy_score

def parzen(classes,x_train,y_train,x_test,y_test):
    
    # y_test=pd.DataFrame(y_test)
    # y_test = pd.factorize(y_test[0])[0]
    # y_train=pd.DataFrame(y_train)
    # y_train = pd.factorize(y_train[0])[0]
    
    # separa 20% para validação do h
    x_train1, x_valid, y_train1, y_valid = train_test_split(x_train, y_train, train_size = 0.8)
    
    # VALIDAÇÃO DO h
    score_val=[]
    h_list=[]
    for h in np.arange(0.1,3,0.05):
    #for h in np.arange(0.1,3,0.5):
        
        # Matriz que armazena as probabilidades para cada classe
        P = pd.DataFrame(data=np.zeros((x_valid.shape[0], len(classes))), columns = classes) 
        Pc = np.zeros(len(classes)) # Armaze a fracao de elementos em cada classe
        for i in np.arange(0, len(classes)): # Para cada classe
            elements = tuple(np.where(y_train1 == classes[i])) # elmentos na classe i
            Pc[i] = len(elements)/len(y_train1) # Probabilidade pertencer a classe i
            Z = x_train1[elements,:][0] # Elementos no conjunto de treinamento
            kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(Z)
            for j in np.arange(0,x_valid.shape[0]): # para cada observacao no conjunto de teste
                x = x_valid[j,:]
                x = x.reshape((1,len(x)))
                # calcula a probabilidade pertencer a cada classe
                pj = np.exp(kde.score_samples(x)) 
                P[classes[i]][j] = pj*Pc[i]
                
        y_pred = [] # Vetor com as classes preditas
        for i in np.arange(0, x_valid.shape[0]):
            c = np.argmax(np.array(P.iloc[[i]]))
            y_pred.append(classes[c])
        #y_pred = np.array(y_pred, dtype=str)
        # calcula a acuracia
        score = accuracy_score(y_pred, y_valid)
        score_val.append(score)
        h_list.append(h)
            
    # plt.title ("Validação do h")
    # plt.xlabel('valor de h')
    # plt.ylabel('score')
    # plt.plot (h_list,score_val)
    # plt.show()
    h_val=h_list[score_val.index(np.max(score_val))]
    print ("h=",h_val)
    # Treinamento e teste
    h=h_val 
    # Matriz que armazena as probabilidades para cada classe
    P = pd.DataFrame(data=np.zeros((x_test.shape[0], len(classes))), columns = classes) 
    Pc = np.zeros(len(classes)) # Armaze a fracao de elementos em cada classe
    for i in np.arange(0, len(classes)): # Para cada classe
        elements = tuple(np.where(y_train == classes[i])) # elmentos na classe i
        Pc[i] = len(elements)/len(y_train) # Probabilidade pertencer a classe i
        Z = x_train[elements,:][0] # Elementos no conjunto de treinamento
        kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(Z)
        for j in np.arange(0,x_test.shape[0]): # para cada observacao no conjunto de teste
            x = x_test[j,:]
            x = x.reshape((1,len(x)))
            # calcula a probabilidade pertencer a cada classe
            pj = np.exp(kde.score_samples(x)) 
            P[classes[i]][j] = pj*Pc[i]
            
    y_pred = [] # Vetor com as classes preditas
    for i in np.arange(0, x_test.shape[0]):
        c = np.argmax(np.array(P.iloc[[i]]))
        y_pred.append(classes[c])
    #y_pred = np.array(y_pred, dtype=str)
    # calcula a acuracia
    #score = accuracy_score(y_pred, y_test)

    return y_test,y_pred, h_list,score_val