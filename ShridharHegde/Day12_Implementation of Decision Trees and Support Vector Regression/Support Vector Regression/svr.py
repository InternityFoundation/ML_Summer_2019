import random
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR 
from sklearn.metrics import mean_squared_error 

#generating data for the model
random.seed(1998)
def getData(N):
 x,y = [],[]
 for i in range(N):  
  a = i/10+random.uniform(-1,1)
  yy = math.cos(a)+random.uniform(-1,1)
  x.append([a])
  y.append([yy])  
 return np.array(x), np.array(y)


#Getting the data
x,y = getData(200)


#Model description
model = SVR()
print(model)


#Train the model for the data
model.fit(x,y)
pred_y = model.predict(x)


#Plot the results
x_axis=range(200)
plt.scatter(x_axis, y, s=5, color="blue", label="original")
plt.plot(x_axis, pred_y, lw=1.5, color="red", label="predicted")
plt.legend()
plt.show() 


score=model.score(x,y)
print(score)
mse =mean_squared_error(y, pred_y)
print("Mean Squared Error:",mse)
rmse = math.sqrt(mse)
print("Root Mean Squared Error:", rmse)
