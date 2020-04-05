"""
Author:         David Walshe
Date:           05/04/2020
Decs:           Linear Discriminant Analysis
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
dataset = pd.read_csv("9_dimensionality_reduction/lda/Wine.csv")
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, -1].values.reshape(-1, 1)

# Splitting the dataset into a training set and a test set
# ========================================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.20, random_state=0
)

# Feature Scaling
# Must be done in PCA & LDA techniques.
# ===============
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Applying LDA
# ============
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
# include y_train as lda is supervised i.e. involves dependant variable.
X_train = lda.fit_transform(X_train, y_train)
# transform only needs independent variables, no fit involved.
X_test = lda.transform(X_test)

# Fitting Logistic Regression to the Training set
# ===============================================
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
# ===============================
y_pred = classifier.predict(X_test).reshape(-1, 1)

# Making the Confusion Matrix
# ===========================
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# Visualising the training set results
from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(("red", "green", "yellow")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

y_set = y_set.flatten()
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(("blue", "orange", "cyan"))(i), label=j)

plt.title("Logistic Regression with LDA (Training set)")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.legend()
plt.show()

# Visualising the test set results
from matplotlib.colors import ListedColormap

X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(("red", "green", "yellow")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

y_set = y_set.flatten()
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(("blue", "orange", "cyan"))(i), label=j)

plt.title("Logistic Regression with LDA (Test set)")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.legend()
plt.show()