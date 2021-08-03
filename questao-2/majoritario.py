# Classificador pelo voto majoritário
import statistics

def majoritario(classes,ypreds,ytest):
    
    
    ypred=[]    
    n=len(ypreds[0])
    for i in range(n):
        ypred.append(statistics.mode([ypreds[0][i],ypreds[1][i],ypreds[2][i],ypreds[3][i]]))
    
    return ytest,ypred