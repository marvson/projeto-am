import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os.path

import data_analysis

# IMPORT DATASET WITH PANDAS
DF_PATH = os.path.dirname(__file__) + "/yeast_csv.csv"
df = pd.read_csv(DF_PATH, encoding="utf-8")

# DATA EXPLORATORY ANALYSIS AND PREPROCESSING
data_analysis.run(df)