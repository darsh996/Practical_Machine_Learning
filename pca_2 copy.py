import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

br = pd.read_csv("Bankruptcy.csv", index_col=0)

X = br.drop(['D','YR'], axis=1)
y = br['D']
X_train, X_test, y_train, y_test = train_test_split(X,y, 
                               test_size=0.3,
                               stratify=y,
                               random_state=23)
scaler = StandardScaler()
prcomp = PCA()

pipe = Pipeline([('SCL',scaler),('PCA', prcomp)])
comps = pipe.fit_transform(X_train)

cum_sum = np.cumsum(prcomp.explained_variance_ratio_*100)

### First 10 components explain more than 95% var
X_trn_pc = comps[:,:10]
X_tst_pc = pipe.transform(X_test)[:,:10]

lr = LogisticRegression()
lr.fit(X_trn_pc, y_train)
y_pred = lr.predict(X_tst_pc)
y_pred_prob = lr.predict_proba(X_tst_pc)
print(log_loss(y_test, y_pred_prob))

################# Entire Pipeline #########################
prcomp = PCA(n_components=0.75)
pipe = Pipeline([('SCL',scaler),('PCA', prcomp),('LR', lr)])

pipe.fit(X_train, y_train)
y_pred_prob = pipe.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

########## Grid Search #############
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
print(pipe.get_params())
params = {'PCA__n_components':[5,6,7,8,9,10]}
# OR
params = {'PCA__n_components':[0.7, 0.75, 0.8, 0.85, 0.9, 0.95]}
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold,
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
