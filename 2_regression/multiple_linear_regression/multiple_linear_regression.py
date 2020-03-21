"""
Author:         David Walshe
Date:           19/03/2020   
"""

# Import libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# Change default output resolutuon of plt.show() to 1920x1440
mpl.rcParams['figure.dpi'] = 300

# import dataset
# ==============
dataset = pd.read_csv("./2_regression/multiple_linear_regression/50_Startups.csv")
# Select all rows, select all columns except the last.
x = dataset.iloc[:, :-1].values
# Select all rows, select only the 4th column.
y = dataset.iloc[:, 4].values.reshape(-1, 1)

# Catergorical Encoding
# =====================
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Create Transformer
ct = ColumnTransformer(
    [("one_hot_encoder", OneHotEncoder(), [3])],
    remainder="passthrough"
)

# Run the transformer on X.
x = np.array(ct.fit_transform(x), dtype=np.float64)

# Split training and test sets
# ============================
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression to the Training set
# ======================================================
from sklearn.linear_model import LinearRegression

# Create a regressor and fit it to the training set.
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
# ===============================
y_pred = regressor.predict(x_test)

# Using backward elimination to generate a better model.
# ======================================================
import statsmodels.formula.api as sm

# Add an array of ones to the start of X to account for constant, C0 in multiple linear regression equation.
# Y = C0 + C1*X1 + C2*X2 + ... + Cn*Xn
x = np.append(arr=np.ones((50, 1)).astype(int), values=x, axis=1)
