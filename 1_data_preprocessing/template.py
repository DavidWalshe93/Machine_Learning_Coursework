"""
Author:         David Walshe
Date:           19/03/2020   
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv("./1_data_preprocessing/Data.csv")

# Select all rows, select all columns except the last.
x = dataset.iloc[:, :-1].values
