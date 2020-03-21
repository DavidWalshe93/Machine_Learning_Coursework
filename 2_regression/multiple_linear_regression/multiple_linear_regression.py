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

# Avoid the dummy variable trap
# =============================
x = x[:, 1:]

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

# Decide a significance level.
sig_level = 0.05


# Phase 1
# =======
# The initial optimised table of features.
x_opt = x[:, [0, 1, 2, 3, 4, 5]]

# Ordinarily Least Squares Regressor.
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
# regressor_OLS.summary()


# Phase 2
# =======
# Removed least significant P-value [2] found in Phase 1.
x_opt = x[:, [0, 1, 3, 4, 5]]

# Ordinarily Least Squares Regressor.
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()


# Phase 3
# =======
# Removed least significant P-value [1] found in Phase 1.
x_opt = x[:, [0, 3, 4, 5]]

# Ordinarily Least Squares Regressor.
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()


# Phase 4
# =======
# Removed least significant P-value [4] found in Phase 1.
x_opt = x[:, [0, 3, 5]]

# Ordinarily Least Squares Regressor.
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

# Phase 5
# =======
# Removed least significant P-value [5] found in Phase 1.
x_opt = x[:, [0, 3]]

# Ordinarily Least Squares Regressor.
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()