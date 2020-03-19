"""
Author:         David Walshe
Date:           19/03/2020   
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Takes care of missing data.
from sklearn.impute import SimpleImputer

# import dataset
# ==============
dataset = pd.read_csv("./1_data_preprocessing/Data.csv")
# Select all rows, select all columns except the last.
x = dataset.iloc[:, :-1].values
# Select all rows, select only the 4th column.
y = dataset.iloc[:, 3].values

# Handle missing data.
# ====================
imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
# Fit the imputer to the size to of the dataset of interest.
imp_mean = imp_mean.fit(x[:, 1:3])
# Fill the missing data and reassign data to original dataset.
x[:, 1:3] = imp_mean.transform(x[:, 1:3])


