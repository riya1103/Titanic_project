import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import re

%matplotlib inline
df=pd.read_csv("train.csv")

df.Pclass.value_counts()
df.Cabin.value_counts()
df.Pclass.unique()

df.describe()
corr=df.corr()
sbn.heatmap(corr,annot=True,xticklabels=corr.columns.values
            ,yticklabels=corr.columns.values)
df.columns
continuous=[ 'Age', 'SibSp',
       'Parch', 'Fare']
df[continuous].plot.box()
df[['Parch','SibSp']].plot.box()
df[['Age']].plot.box()

df[['Fare']].plot.box()
x=df['Fare'].max()
df['Fare'].count()
df[df['Fare']>263].Survived.count()
df[df['Fare']>75].Survived.sum()/df[df['Fare']>75].Survived.count()
df.Survived.sum()/df.Survived.count()
#important
fare=[]
survived=[]
out =pd.DataFrame()

for i in range(0,513):
    survived.append(df[df['Fare']>i].Survived.sum()/df[df['Fare']>i].Survived.count())
    fare.append(i)

out['fare above']=fare
out['survival rate']=survived

out.to_clipboard()




df.plot()
plt.hist(df['Pclass'],bins=10,xlabel='Pclass')
plt.hist(df['Cabin'],bins=10)
#plt.hist(df['Sex'],bins=10)
plt.hist(df['SibSp'],bins=10)
plt.hist(df['Fare'],bins=10)
plt.hist(df['Parch'],bins=10)
df.info('Pclass')
df.describe(include='all')
pd.crosstab(df['Cabin'])
my_tab = pd.crosstab(index = df["Cabin"],  # Make a crosstab
                              columns="count")      # Name the count column

my_tab.plot.bar()

my_tab = pd.crosstab(index = df["Embarked"],  # Make a crosstab
                              columns="count")      # Name the count column

my_tab.plot.bar()


my_tab = pd.crosstab(index = df["Sex"],  # Make a crosstab
                              columns="count")      # Name the count column

my_tab.plot.bar()


my_tab = pd.crosstab(index = df["Ticket"],  # Make a crosstab
                              columns="count")      # Name the count column

my_tab.plot.bar()

my_tab = pd.crosstab(index = df["Pclass"],  # Make a crosstab
                              columns="count")      # Name the count column

my_tab.plot.bar()

df['Relatives']=df['Parch']+df['SibSp']
plt.hist(df['Relatives'],bins=10)


df[df['Sex']=='male'].Survived.count() #it is simply counting the non null values.

df[df['Sex']=='female'].Survived.count()

#df[[df['Sex']=='female' and df['Survived']==0]].count() check it!!

#Survival rates of the categorical variables

sr_mgender=df[df['Sex']=='male'].Survived.sum()/df[df['Sex']=='male'].Survived.count()

sr_fgender=df[df['Sex']=='female'].Survived.sum()/df[df['Sex']=='female'].Survived.count()


sr_1pclass=df[df['Pclass']==1].Survived.sum()/df[df['Pclass']==1].Survived.count()

sr_2pclass=df[df['Pclass']==2].Survived.sum()/df[df['Pclass']==2].Survived.count()

sr_3pclass=df[df['Pclass']==3].Survived.sum()/df[df['Pclass']==3].Survived.count()

sr_1pclass=df[df['Pclass']==1].Survived.sum()/df[df['Pclass']==1].Survived.count()


sr_Cembark=df[df['Embarked']=='C'].Survived.sum()/df[df['Embarked']=='C'].Survived.count()

sr_Qembark=df[df['Embarked']=='Q'].Survived.sum()/df[df['Embarked']=='Q'].Survived.count()

sr_Sembark=df[df['Embarked']=='S'].Survived.sum()/df[df['Embarked']=='S'].Survived.count()

#using groupby() 
u= df.groupby(['Embarked']).Survived.count()

v= df.groupby(['Embarked']).Survived.sum()
sr_embarked=(v/u)*100

sr_gender=df.groupby(['Sex']).Survived.sum()/df.groupby(['Sex']).Survived.count()

sr_Pclass=df.groupby(['Pclass']).Survived.sum()/df.groupby(['Pclass']).Survived.count()

df['Age'].max()
df['Age'].min()

age=[]
survived_age=[]
out_age=pd.DataFrame()
for i in range(0,81):
    survived_age.append(df[df['Age']]>i).Survived.sum()/(df[df['Age']]>i).Survived.count()
    age.append(i)
    
out_age['age above']=age
out_age['survival rate']=survived_age


out_age.to_clipboard() 

#plot with survived
my_tab = pd.crosstab(index = df["Pclass"],  # Make a crosstab
                              columns=df["Survived"])      # Name the count column

my_tab.plot.bar()

my_tab = pd.crosstab(index = df["Sex"],  # Make a crosstab
                              columns=df["Survived"] )      # Name the count column

my_tab.plot.bar()
my_tab = pd.crosstab(index = df["Embarked"],  # Make a crosstab
                              columns=df['Survived'])      # Name the count column

my_tab.plot.bar()

pd.crosstab(index = df["Embarked"], columns=df['Survived']).plot.bar()

pd.crosstab(index = df["Fare"], columns=df["Survived"]).plot.hist()

df.plot.scatter(x='Sex',y='Survived')

#plotting with seaborn

sbn.kdeplot(df['Fare'])

sbn.kdeplot(df['Age'])

sbn.kdeplot(df['Relatives'],df['Survived'],kernel='gau')

sbn.violinplot(df['Relatives'],df['Survived'])

rx=re.compile("[^\W\d_]+", re.UNICODE)
rx.findall(df['Cabin'])

x=df['Cabin']

df['Relatives'].max()
df['Relatives'].count()
df.groupby(['Relatives'])
df['Relatives'].unique()
#exact number of relatives survival rate

rel=[]
surv_rel=[]
out_rel=pd.DataFrame()

    
for i in [1,0,4,2,6,5,3,7,10]  :
    surv_rel.append(df[df['Relatives']=i].Survived.sum()/df[df['Relatives']==i].Survived.count())
    rel.append(i)

out_rel['Relatives ']=rel
out_rel['survival rate']=surv_rel

out_rel.to_clipboard()

df.groupby(['Relatives']).Survived.mean()

df.groupby(['Relatives']).Survived.count()

from sklearn.metrics import confusion_matrix
cm = 
df['Relatives']=df['Parch']+df['SibSp']
df.columns




df.corr()
df["Age"].coeff()
#matrix of independent features X without the cabin and the ticket and the name column
X=df.iloc[:,[2,4,5,9,11,12]]

#encoding the categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[["Sex","Pclass","Embarked"]] = labelencoder_X.fit.transform(X[["Sex","Pclass","Embarked"]])
onehotencoder = OneHotEncoder(categorical_features = X[["Sex","Pclass","Embarked"]])
X = onehotencoder.fit_transform(X).toarray()

from sklearn.preprocessing import Imputer
imputer= Imputer(missing_values="NaN", strategy="mean",axis=0)
imputer=imputer.fit(X[:,[2,3]])
X[:,[2,3]]= imputer.transform((X[:,[2,3]]))



X=df[["Sex",'Pclass','Age','Fare','Embarked','Relatives']]#excluding cabin and ticket
Y=df["Survived"]
#different approach
X.dtypes
X.Pclass=X.Pclass.astype(object)
X=pd.DataFrame(X["Pclass"],dtype='category')#pd.DataFrame is used to 'CREATE' a df

X=pd.get_dummies(X)

X.fillna(X.mean(),inplace=True)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X,Y)

df2=pd.read_csv("test.csv")
df2.columns
df2=df2.drop(['PassengerId', 'Name','Ticket',  'Cabin'], axis=1)
df2['Relatives']=df2['Parch']+df2['SibSp']
x=df2[['Sex','Pclass','Age','Fare','Embarked','Relatives']]

x.Pclass=x.Pclass.astype(object)
x=pd.get_dummies(x)
x.fillna(X.mean(),inplace=True)
y_pred=classifier.predict(x)





