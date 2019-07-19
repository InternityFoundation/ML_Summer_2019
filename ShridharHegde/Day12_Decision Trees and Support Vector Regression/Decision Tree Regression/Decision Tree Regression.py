#Import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor

#Importing the dataset
datasets = pd.read_csv('salaries.csv')
X = datasets.iloc[:, 1:2].values
Y = datasets.iloc[:, 2].values

# Fitting the Regression model to the dataset
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,Y)

# Visualising the Decision Tree Regression results in higher resolution and smoother curve
X_Grid = np.arange(min(X), max(X), 0.01)
X_Grid = X_Grid.reshape((len(X_Grid), 1))
plt.scatter(X,Y, color = 'red')
plt.plot(X_Grid, regressor.predict(X_Grid), color = 'blue')
plt.title('Decision Tree Regression Results')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()