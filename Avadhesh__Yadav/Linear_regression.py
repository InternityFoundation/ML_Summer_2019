import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
data = pd.read_csv(r'G:\salary_data.csv')
X=data.iloc[:,0:1].values
Y=data.iloc[:,1:].values
mean_x = np.mean(X)
mean_Y = np.mean(Y)
m = len(X)
numer=0
denom=0
for i in range(m):
    numer+=(X[i]-mean_x)*(Y[i]-mean_Y)
    denom+=(X[i]-mean_x)**2
b1=numer/denom
b0=mean_Y - (b1*mean_x)
print(b1,b0)
max_x=np.max(X)+100
min_x=np.min(X)-100
x=np.linspace(min_x,max_x,1000)
y=b0+b1*x
plt.plot(x,y,color='#58b970',label='Regression Line')
plt.scatter(x,y,color='#ef5423',label='scatter plot')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()