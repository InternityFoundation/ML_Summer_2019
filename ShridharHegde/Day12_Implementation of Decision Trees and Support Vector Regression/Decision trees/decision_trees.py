import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.metrics import classification_report, confusion_matrix 

#Read the data and get informationa about it.
data = pd.read_csv("bill.csv")  
print(data.shape)
print(data.head())

#Dividing data into values and labels
X = data.drop('Class', axis=1)  
y = data['Class'] 

#Splitting the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)   

#Decision Tree Classifier
classifier = DecisionTreeClassifier()  
classifier.fit(X_train, y_train)

#Run the model on the test data
y_pred = classifier.predict(X_test)  
 
#Obtaining the accuracy of the model
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  
