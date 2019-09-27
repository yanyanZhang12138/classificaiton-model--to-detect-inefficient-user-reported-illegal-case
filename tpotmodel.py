import numpy as np 
import pandas as pd 
from sklearn import preprocessing 
from sklearn.metrics import mean_squared_error 
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb 

dfee0 = pd.read_csv("dfee0new.csv")
df = dfee0
dfee0['status'] = dfee0['status'].astype(int)

target = 'status'
IDcol = 'uid'
predictors = [x for x in df.columns if x not in [target, IDcol]]
X_train, X_test, y_train, y_test = train_test_split(dfee0[predictors], dfee0[target], train_size = 0.75, test_size = 0.25)

tpot = TPOTClassifier(generations = 100, population_size = 100, verbosity = 2, n_jobs = 8)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_report_pipeline.py')

