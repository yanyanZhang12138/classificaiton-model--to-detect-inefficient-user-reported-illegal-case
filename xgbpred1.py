import xgboost as xgb 
import pandas as pd 
import numpy as np
from sklearn import svm
from xgboost import XGBClassifier
import matplotlib.pylab as plt 
from matplotlib.pylab import rcParams
from sklearn.model_selection import cross_validate, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

model = joblib.load('rp_xgbmodel3.model')

dftest=pd.read_table('20190917newreport.txt', delim_whitespace=True, names=('uid', 'people_count', 'user_level', 'fans_count', 'official', 'money', 'live_count', 'diamond', 'content_cnt', 'status', 'punish_rate'))

#%%
def dftest_processing():
    del dftest['official']
    dftest['people_count'] = dftest['people_count'].astype(int)
    dftest["user_level"] = dftest["user_level"].astype(int)
    dftest['fans_count'] = dftest['fans_count'].astype(int)
    dftest['money'] = dftest['money'].astype(int)
    dftest['live_count'] = dftest['live_count'].astype(int)
    dftest['diamond'] = dftest['diamond'].astype(int)
    dftest['content_cnt'] = dftest['content_cnt'].astype(int)
    dftest['punish_rate'] = dftest['punish_rate'].astype(float)
    dftest['uid']=dftest['uid'].astype(object)
    dftest['status']=dftest['status'].astype(str)

#%%
dftest_processing()
print(dftest.head())


target = 'status'
IDcol = 'uid'
predictors = [x for x in dftest.columns if x not in [target, IDcol]]
x_test = dftest[predictors]
dtest = xgb.DMatrix(x_test)
ans = model.predict(dtest)
y_true = dftest[target]
cm1 = pd.crosstab(y_true, ans, rownames = ['Actual'], colnames=['Predicted'])
print(cm1)
pd.DataFrame(ans).to_csv("20190917ans.csv")
