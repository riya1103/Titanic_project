import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
df=pd.read_csv("train.csv")
df['Relatives']=df['Parch']+df['SibSp']

X=df[["Sex",'Pclass','Age','Fare','Embarked','Relatives']]#excluding cabin and ticket
Y=df["Survived"]
#different approach
X.dtypes
X.Pclass=X.Pclass.astype(object)

X.fillna(X.mean(),inplace=True)
X=pd.get_dummies(X, drop_first=True)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0)
classifier.fit(X,Y)

df2=pd.read_csv("test.csv")
df2.columns
#df2=df2.drop(['PassengerId', 'Name','Ticket',  'Cabin'], axis=1)
df2['Relatives']=df2['Parch']+df2['SibSp']
x=df2[['Sex','Pclass','Age','Fare','Embarked','Relatives']]

x.Pclass=x.Pclass.astype(object)
x.fillna(X.mean(),inplace=True)

x=pd.get_dummies(x, drop_first=True)
y_pred=classifier.predict(x)
df3=pd.DataFrame()
df3['PassengerId']=df2['PassengerId']
df3['Survived']=y_pred

df3.to_clipboard()
from xgboost import XGBClassifier
classifier= XGBClassifier(n_estimators=500, max_depth=3)
classifier.fit(X,Y)
y_pred=classifier.predict(x)

df4=pd.DataFrame()
df4['PassengerId']=df2['PassengerId']

df4['Survived']=classifier.predict(x)
df4.to_csv('xgb.csv',index=False)

