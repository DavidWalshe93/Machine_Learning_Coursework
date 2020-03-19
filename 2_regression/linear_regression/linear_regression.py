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
dataset = pd.read_csv("./2_regression/linear_regression/Salary_Data.csv")
# Select all rows, select all columns except the last.
x = dataset.iloc[:, 0].values.reshape(-1, 1)
# Select all rows, select only the 4th column.
y = dataset.iloc[:, 1].values.reshape(-1, 1)


# Splitting the dataset into a training set and a test set
# ========================================================
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=1/3, random_state=0
)


# Fitting Simple Linear Regression to the Training Set
# ====================================================
from sklearn.linear_model import LinearRegression

# Create and fit the training sets.
regressor = LinearRegression().fit(x_train, y_train)


# Predicting the Test set Results
# ===============================
y_pred = regressor.predict(x_test)


# Visualising the Training set Results
# =======================
plt.scatter(x_train, y_train, color="red")
plt.plot(x_train, regressor.predict(x_train), color="blue")
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")

plt.show()

