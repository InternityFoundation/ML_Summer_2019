# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 11:00:20 2019

@author: Santosh
"""

#importing necessary libraries to run the program
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


#load the data and analyze it
data=pd.read_csv("zoo.csv")
print(data.describe())

#split the data into features and target variables
X=data.iloc[:,1:-1]
print(X.describe())
Y=data.iloc[:,-1].values.ravel()

#split the data into training data and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

#fit the training data into the Random Forest Classifier
model=RandomForestClassifier().fit(X_train, Y_train)

#Provide the testing data for the model
predict = model.predict(X_test)

#Compare the predicted class and the actual class
print(predict.ravel())
print(Y_test)

#Computing the efficiecny
print("Efficiency is :")
count=0
for i in range(len(Y_test)):
    if(Y_test[i]==predict[i]):
        count=count+1
        
print((count/len(Y_test))*100)

