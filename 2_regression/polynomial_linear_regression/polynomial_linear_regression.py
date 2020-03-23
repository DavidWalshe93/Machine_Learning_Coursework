"""
Author:         David Walshe
Date:           21/03/2020
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
dataset = pd.read_csv("./2_regression/polynomial_linear_regression/Position_Salaries.csv")
# Select all rows, select all columns except the last.
x = dataset.iloc[:, 1:2].values.reshape(-1, 1)
# Select all rows, select only the 4th column.
y = dataset.iloc[:, 2].values.reshape(-1, 1)


# No test/train set split as dataset is too small


# Fitting Linear Regression to the Dataset ( Reference )
# ======================================================
from sklearn.linear_model import LinearRegression

# Create and fit linear regressor to the dataset
lin_reg = LinearRegression().fit(x, y)


# Fitting Polynomial Regression to the Dataset.
# =============================================
from sklearn.preprocessing import PolynomialFeatures

# Convert the X feature table into a table of 4 COLs with [1, X, X^2, X^3]
poly_reg = PolynomialFeatures(degree=10)
x_poly = poly_reg.fit_transform(x)

# Create and fit a linear regressor to the polynomial dataset, x_poly
lin_reg_poly = LinearRegression().fit(x_poly, y)


# Visualising the Linear Regression Results
# =========================================
plt.scatter(x, y, color="red")
plt.plot(x, lin_reg.predict(x), color="blue")
plt.title("Salary Guide (Linear Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# Visualising the Polynomial Regression Results
# =============================================

# Create higher precision for X axis.
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x, y, color="red")
# Use x_grid in place of X for smoother line plot.
plt.plot(x_grid, lin_reg_poly.predict(poly_reg.fit_transform(x_grid)), color="blue")
plt.title("Salary Guide (Polynomial Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()


# Predicting a new result with Linear Regression
# ==============================================
print(f"Linear Regression Results:\t\t{lin_reg.predict([[6.5]])}")

# Predicting a new result with Polynomial Regression
# ==================================================
print(f"Polynomial Regression Results:\t{lin_reg_poly.predict(poly_reg.fit_transform([[6.5]]))}")