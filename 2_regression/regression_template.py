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
x = dataset.iloc[:, 1:2].values.reshape(-1, 1)
y = dataset.iloc[:, 2].values.reshape(-1, 1)
