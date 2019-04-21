import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import seaborn as sbn

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


