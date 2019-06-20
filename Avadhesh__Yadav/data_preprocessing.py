import pandas as pd
from sklearn.preprocessing import Imputer
data={'country':['India','India','Pakistan','srilanka','Bangladesh','India','Pakistan'],'age':[21,'Nan',25,30,'Nan',24,29],'salary':[5000,6000,2000,'NaN',1500,'Nan',1800],'purchased':['yes','no','yes','no','no','yes','no']}
df=pd.DataFrame(data)
X=df.iloc[:,0:3].values
Y=df.iloc[:,3:].values
imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
imp=imp.fit(X[:,1:3])
X[:,1:3]=imp.transform(X[:,1:3])
print(X)
