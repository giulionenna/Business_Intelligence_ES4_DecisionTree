
import pandas as pd
import numpy as np
from sklearn import tree
import sklearn
from IPython.display import Image  
import pydotplus
import graphviz
import matplotlib.pyplot as plt
data = pd.read_excel(r'C:\Users\giuli\Documents\GitHub\Business_Intelligence_ES4_DecisionTree\Users.xls', na_values='?')
df = pd.DataFrame(data)
df[df['Workclass'].isna()]

df = df.fillna(df.mode().iloc[0])
df[df['Workclass'].isna()]
df.dtypes
df_enc = df

for col in df_enc.columns:
    df_enc[col]=df_enc[col].astype('category')
    df_enc[col]=df_enc[col].cat.codes

df_enc['Age']=df_enc['Age'].astype('int64')


dfX= df_enc.iloc[:, 0:9]
dfY = df_enc.iloc[:, 9]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(dfX, dfY)

tree.plot_tree(clf)
plt.figure()
df.plot()
plt.show()