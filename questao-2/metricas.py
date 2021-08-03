import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np

def metricas(test,pred,classes):
    
    # test=pd.DataFrame(test)
    # test = pd.factorize(test[0])[0]
    # pred=pd.DataFrame(pred)
    # pred = pd.factorize(pred[0])[0]
    
    err = 1-accuracy_score(test,pred)
    pre = precision_score(test, pred, average='weighted', zero_division=0)
    rec = recall_score(test, pred, average='weighted', zero_division=0)
    fsc = f1_score(test, pred, average='weighted', zero_division=0)
        #cr=classification_report(test, pred, target_names=classes)
        
    
    
    return err,pre,rec,fsc

def confint(n,err,pre,rec,fsc):
    
    const = 1.96  # intervalo de confiança de 95%
    confint_err = const * np.sqrt( (err * (1 - err)) / n)
    confint_pre = const * np.sqrt( (pre * (1 - pre)) / n)
    confint_rec = const * np.sqrt( (rec * (1 - rec)) / n)
    confint_fsc = const * np.sqrt( (fsc * (1 - fsc)) / n)
    
    return confint_err,confint_pre,confint_rec,confint_fsc