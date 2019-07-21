# Step 1: Preprocessing
#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing data
dataset = pd.read_csv("train.csv")
X = dataset.iloc[:,2:].values
Y = dataset.iloc[:,1].values

# split into training n test
from sklearn.model_selection import train_test_split
X_train,  X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=3558)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10)
regressor.fit(X_train,Y_train)

Y_pred = regressor.predict(X_test)

