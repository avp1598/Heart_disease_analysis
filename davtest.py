from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(0)
data = pd.read_csv('D:\\Coding\\data_science\\Tests\\DAV\\heart.csv')
#print(data.head(5))
#data.groupby('target')['cp'].value_counts().plot.bar()
#temp=data[data['target']==0]
#temp['cp'].value_counts().plot.bar(rot=0)
#data.hist(column='age',bins=10,grid=False)
#print(temp.head(5))
#temp.boxplot(column='age',grid=False)
#data['cp'].value_counts().plot.bar()
#plt.show()
#cr=data.corr()
#print (cr['target'].sort_values(ascending=False), '\n')
#ax = sns.heatmap(cr,vmin=-1,vmax=1)


data.drop(['fbs','chol'],axis=1)
y=data['target']
X=data.drop(['target'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22)
'''
param_grid = { 
    'n_estimators': list(range(100,500,20)),
    'max_depth' : list(range(30,100,5))
}
clf=RandomForestClassifier()
CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)
print(CV_rfc.best_params_)

param_grid = { 
    'n_neighbors': list(range(50,150))
}
knn=KNeighborsClassifier()
CV_rfc = GridSearchCV(estimator=knn, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)
print(CV_rfc.best_params_)
'''
clf=RandomForestClassifier(n_estimators=16,max_depth=9)
clf.fit(X_train, y_train)
scores = cross_val_score(clf, X_train, y_train, cv=5)
print(scores.mean())

knn=KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train, y_train)
scores = cross_val_score(knn, X_train, y_train, cv=5)
print(scores.mean())
#'''
