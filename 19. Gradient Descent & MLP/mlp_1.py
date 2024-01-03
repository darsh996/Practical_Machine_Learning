import pandas as pd 
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss,r2_score
from sklearn.model_selection import KFold
import numpy as np
hr = pd.read_csv("/Users/darshmac/Documents/cdac/For DBDA/Advance Analytics/Datasets/Boston.csv")
hr
#dum_hr = pd.get_dummies(hr, drop_first=True)
X = hr.drop('medv', axis=1)
y = hr['medv'].values

scl_x = MinMaxScaler()
scl_y = MinMaxScaler() 

scaled_y = scl_y.fit_transform(y.reshape(-1,1))


mlp = MLPRegressor(random_state=23, 
                    hidden_layer_sizes=(7,5,4,3),
                    activation='relu')

pipe = Pipeline([('SCL',scl_x),('MLR',mlp)])

X_train,X_test,y_train,y_test = train_test_split(X,scaled_y[:,0],test_size=0.3,random_state=23)

pipe.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
#print(accuracy_score(y_test, y_pred))
print(r2_score(y_test, y_pred))

y_pred_prob = mlp.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

############# Grid Search CV #####################
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import StratifiedKFold
print(mlp.get_params())

#MLP Regressor
mlp = MLPRegressor(random_state=23, 
                    hidden_layer_sizes=(7,5,4,3),
                    activation='logistic')

kfold = KFold(n_splits=5, shuffle=True, random_state=23)
params = {'learning_rate':['invscaling','adaptive','constant'],
          'learning_rate_init': [0.1, 0.2, 0.4],
          'hidden_layer_sizes':[(10, 7), (7,5,4), (10, 5)]}

gcv_mlp = GridSearchCV(mlp, param_grid=params,verbose=3,
                       cv=kfold, scoring='r2')
gcv_mlp.fit(X_train, y_train)

print(r2_score(y_test, y_pred

