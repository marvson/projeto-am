from pandas.core.tools import numeric
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path

# IMPORT DATASET WITH PANDAS
DF_PATH = os.path.dirname(__file__) + "/../data/yeast_csv.csv"
df = pd.read_csv(DF_PATH, encoding="utf-8")

def check_missing_values(df):
    df.info()
    missing = np.sum(df.isnull().sum(axis=0))
    print("Total missing values: ", missing)
    return missing

def check_categorical(df, apply_one_hot=True):
    X = df.iloc[: , :-1]
    cat = X.select_dtypes(exclude=["float", 'int'])
    if len(list(cat.columns)) == 0:
        print("The dataset has no categorical attributes")
        return 0
    else:
        print("Dataset categorical attributes: \n", list(cat.columns))
        if apply_one_hot == True:
            print("Applying One-Hot Encoding")
            for column in list(cat.columns):
                df = df.join(pd.get_dummies(df[column]))
                df = df.drop(columns=[column])  
    print("New dataset attributes: \n ", list(df.columns))
    return df

def check_class_balance(df):
    labels = list(df.iloc[:,-1:].columns)[0]
    sns.catplot(x=labels, kind="count", palette="ch:.25", data=df)
    plt.show()

def check_attributes_distribution(df):
    print(df.describe())

    X = df.iloc[: , :-1]
    X_num = X.select_dtypes(include=["float", 'int'])
    
    for column in list(X_num.columns):
        sns.displot(X_num, x=column, kind="kde", fill=True)
    plt.show()

def check_correlation(df):
    sns.heatmap(df.corr("pearson"))
    sns.pairplot(df.sample(1000), diag_kind='kde')
    plt.show()

check_correlation(df)