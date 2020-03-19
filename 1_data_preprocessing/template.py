"""
Author:         David Walshe
Date:           19/03/2020   
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
# ==============
dataset = pd.read_csv("./1_data_preprocessing/Data.csv")
# Select all rows, select all columns except the last.
x = dataset.iloc[:, :-1].values
# Select all rows, select only the 4th column.
y = dataset.iloc[:, 3].values.reshape(-1, 1)

# Takes care of missing data.
from sklearn.impute import SimpleImputer

# Handle missing data.
# ====================
imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
# Fit the imputer to the size to of the dataset of interest.
imp_mean = imp_mean.fit(x[:, 1:3])
# Fill the missing data and reassign data to original dataset.
x[:, 1:3] = imp_mean.transform(x[:, 1:3])

# Encoding categorical data.
# ==========================
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Deprecated version of handling categorical data.
# ************************************************
# le_x = LabelEncoder()
# # Fit the label encoder to the data of interest.
# le_x = le_x.fit(x[:, 0])
# # Mask the categorical data of the first column as integer numerical.
# x[:, 0] = le_x.transform(x[:, 0])
#
# ohe = OneHotEncoder(categorical_features=[0])
# x = ohe.fit_transform(x).toarray()


# Newer way of converting Categorical Data
# ****************************************
# Create a column transformer using a OneHotEncoder to convert
# Categorical Data to Binary Encoded data.
ct = ColumnTransformer(
    [("one_hot_encoder", OneHotEncoder(), [0])],
    remainder="passthrough"
)

# Convert X[:, 0] categories to binary encoding.
x = np.array(ct.fit_transform(x), dtype=np.int32)
# Convert Y category (Yes/No) to binary encoding.
y = np.array(ct.fit_transform(y), dtype=np.int32)

# Splitting the dataset into a training set and a test set
# ========================================================
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=0
)

# Feature scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

# Scale the x_train, x_test datasets.
x_train[:, 3:] = sc_X.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc_X.transform(x_test[:, 3:])  # Dont re-fit as X_train cols == X_test cols
