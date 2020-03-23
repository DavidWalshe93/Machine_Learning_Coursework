"""
Author:         David Walshe
Date:           21/03/2020   
"""

# Import libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# Change default output resolutuon of plt.show() to 1920X1440
mpl.rcParams['figure.dpi'] = 300

# import dataset
# ==============
dataset = pd.read_csv("2_regression/decision_tree_regression/Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values.reshape(-1, 1)
y = dataset.iloc[:, 2].values.reshape(-1, 1)

# # Feature Scaling required for Decision Tree
# # ================================
# from sklearn.preprocessing import StandardScaler
# # Scale X
# sc_X = StandardScaler()
# X = sc_X.fit_transform(X)
# # Scale y
# sc_y = StandardScaler()
# y = sc_y.fit_transform(y)

# # Fitting  to the dataset
# # ==========================
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)


# Predicting the new result
# =========================
y_pred = regressor.predict([[6.5]])
# temp_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))
# y_pred = sc_y.inverse_transform(temp_pred)


# Visualising the Decision Tree results
# ===========================
# Create higher precision for X axis.
x_grid = np.arange(min(X), max(X), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(X, y, color="red")
# Use x_grid in place of X for smoother line plot.
plt.plot(x_grid, regressor.predict(x_grid), color="blue")
plt.title("Salary Guide (Decision Tree Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()