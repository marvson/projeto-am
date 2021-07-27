import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman

from bayesiano import bayesiano
from knn import knn
from parzen import parzen
from logistic import logistic
from majoritario import majoritario

from metricas import metricas
from metricas import confint

    
data = pd.read_csv('yeast.data', header=(0))
data = data.drop(columns=["a1"])
#classes = np.array(pd.unique(data[data.columns[-1]]), dtype=str)
data = data.to_numpy()
nrow,ncol = data.shape
y = data[:,-1]
x = data[:,0:ncol-1]

# Transforma os dados para terem media igual a zero e variancia igual a 1
scaler = StandardScaler().fit(x)
x = scaler.transform(x)
# X = X.astype('float64')

y = pd.factorize(y)[0]
classes = pd.factorize(y)[1]
classes = pd.factorize(classes)[0]

# prepara validação cruzada k-fold estratificada
kfold = StratifiedKFold(n_splits=5, shuffle=True)

ypreds=[]
results=[]
nome=["Bayesiano","K-nn bayesiano","Bayesiano janela Parzen","Regressão logística","Voto Majoritário"]

for j in range(5):
    erro=[]
    prec=[]
    reca=[]
    fsco=[]
    f=0
    for train, test in kfold.split(x,y):
        if j==0: ytest,ypred = bayesiano(classes,x[train],y[train],x[test],y[test])
        if j==1: ytest,ypred = knn(classes,x[train],y[train],x[test],y[test])
        if j==2: ytest,ypred = parzen(classes,x[train],y[train],x[test],y[test])
        if j==3: ytest,ypred = logistic(classes,x[train],y[train],x[test],y[test])
        if j==4: ytest,ypred = majoritario(classes,[ypreds[0+f],ypreds[5+f],ypreds[10+f],ypreds[15+f]],y[test])
        f=f+1
        (err,pre,rec,fsc)=metricas(ytest,ypred,classes)
        erro.append(err)
        prec.append(pre)
        reca.append(rec)
        fsco.append(fsc)
        ypreds.append(ypred)
    results.append(ypred)
    erro=np.mean(erro)
    prec=np.mean(prec)
    reca=np.mean(reca)
    fsco=np.mean(fsco)
    (confint_err,confint_pre,confint_rec,confint_fsc)=confint(len(ytest),erro,prec,reca,fsco)
    print("\n Classificador ",nome[j])
    print("    erro:",erro.round(decimals=3)," +/- ",confint_err.round(decimals=3))
    print("precisão:",prec.round(decimals=3)," +/- ",confint_pre.round(decimals=3))
    print("  recall:",reca.round(decimals=3)," +/- ",confint_rec.round(decimals=3))
    print(" F-Score:",fsco.round(decimals=3)," +/- ",confint_fsc.round(decimals=3))

results2=np.array(results)

stat, p = friedmanchisquare(*results2)
print("\n Friedman test (p-value):", p)

if p < 0.05:
    print(f"\n There is significant difference (95% confidence)")
    phn = posthoc_nemenyi_friedman(results2.T)
    print(phn)
else:
    print(f"\n There is no significant difference (95% confidence)")
