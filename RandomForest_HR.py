import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

#%%
hr = pd.read_csv("E:\CDAC\Advanced Analytics\Datasets\Cases\human-resources-analytics\\HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr, drop_first=True)
x = dum_hr.drop('left', axis=1)
y = dum_hr['left']

#%%
rf=RandomForestClassifier(n_estimators=20, random_state=23)
params={'max_features':[2, 4, 6, 8, 10], 'min_samples_split':[2, 5, 20, 80, 100], 'max_depth':[3, 4, 6, 7, None], 'min_samples_leaf':[1, 5, 10, 20]}

kfold=StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
gsv=GridSearchCV(rf, param_grid=params, cv=kfold, scoring='neg_log_loss', verbose=4)
gsv.fit(x, y)

#%%

print(gsv.best_params_)
print(gsv.best_score_)
bm_rf=gsv.best_estimator_

#%%
#params={'max_features':[2, 4, 6, 8, 10]}
#print(gsv.best_params_) #{'max_features': 4}
#print(gsv.best_score_) #-0.09102977847758394

#params={'max_features':[2, 4, 6, 8, 10], 'min_samples_split':[2, 5, 20, 80, 100], 'max_depth':[3, 4, 6, ,7, None], 'max_samples_leaf':[1, 5, 10, 20]}
#print(gsv.best_params_) #{'max_depth': 7, 'max_features': 10, 'min_samples_leaf': 1, 'min_samples_split': 5}
#print(gsv.best_score_) #-0.08116403866117373

#%%     -   Individual Tree
params={'min_samples_split':[2, 5, 20, 80, 100], 'max_depth':[3, 4, 6, 7, None], 'min_samples_leaf':[1, 5, 10, 20]}
dtc=DecisionTreeClassifier(random_state=23)
kfold=StratifiedKFold(n_splits=5, random_state=23, shuffle=True)
gsv=GridSearchCV(dtc, param_grid=params, cv=kfold, scoring='neg_log_loss')
gsv.fit(x, y)

#%%

print(gsv.best_params_)#{'max_depth': 7, 'min_samples_leaf': 5, 'min_samples_split': 100}
print(gsv.best_score_)#-0.09345373740479428
bm_tree=gsv.best_estimator_

#%%     -   Importances plot
#of tree
print(bm_tree.feature_importances_)
df_imp=pd.DataFrame({'Features':list(x.columns), 'Importance':bm_tree.feature_importances_})
df_imp=df_imp[df_imp['Importance']>0].sort_values('Importance')
plt.barh(df_imp['Features'], df_imp['Importance'])
plt.title("Tree")
plt.show()
#%%
#of rf
print(bm_tree.feature_importances_)
df_imp=pd.DataFrame({'Features':list(x.columns), 'Importance':bm_rf.feature_importances_})
df_imp=df_imp[df_imp['Importance']>0].sort_values('Importance')
plt.barh(df_imp['Features'], df_imp['Importance'])
plt.title("Tree")
plt.show()