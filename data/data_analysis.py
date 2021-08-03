from pandas.core.tools import numeric
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path

def run(df):
    """
    Return dataset after one-hot ending, number of missing values and attributes correlation.

    Plots:
    Classes histogram distribution
    Attributes density distribution
    Attributes correlation values (pearson coefficient)
    Attributes pair-correlation

    Parameters
        ----------
    df : pandas dataframe
    """
    missing = check_missing_values(df)
    df = check_categorical(df)
    check_class_balance(df)
    check_attributes_distribution(df)
    corr = check_correlation(df)
    return df, missing, corr

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
        return df
    else:
        print("Dataset categorical attributes: \n", list(cat.columns))
        if apply_one_hot == True:
            print("Applying One-Hot Encoding")
            for column in list(cat.columns):
                X = X.join(pd.get_dummies(df[column]))
                X = X.drop(columns=[column])
    print("New dataset attributes: \n ", list(X.columns))
    df = X.join(df.iloc[: ,-1:])
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
    corr = df.corr("pearson")
    sns.heatmap(corr[(corr >= 0.25) | (corr <= -0.25)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True)
    sns.pairplot(df.sample(1000), diag_kind='kde')
    plt.show()
    return corr