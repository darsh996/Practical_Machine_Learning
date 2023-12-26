import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier 
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

hr = pd.read_csv("HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr, drop_first=True)
X = dum_hr.drop('left', axis=1)
y = dum_hr['left']

X_train, X_test, y_train, y_test = train_test_split(X,y, 
                               test_size=0.3,
                               stratify=y,
                               random_state=23)
lr = LogisticRegression()
svm = SVC(kernel='linear')
dtc = DecisionTreeClassifier(random_state=23)
rf = RandomForestClassifier(random_state=23,
                            n_estimators=25)

stack = StackingClassifier(estimators=[('LR', lr),
                                       ('SVM', svm),
                                       ('TREE', dtc)],
                           final_estimator=rf)
stack.fit(X_train, y_train) # all train set operations
y_pred = stack.predict(X_test) # all test operations
print(accuracy_score(y_test, y_pred))
