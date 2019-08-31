import pandas as pd
import numpy as np
dataset = pd.read_csv('iris.data')
X = dataset.iloc[:,0:4]
y = dataset.iloc[:,4]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Accuracy on training set: {:.2f}'.format(logreg.score(X_train, y_train)))
print('Accuracy on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
print('Accuracy on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy on test set: {:.2f}'.format(clf.score(X_test, y_test)))
