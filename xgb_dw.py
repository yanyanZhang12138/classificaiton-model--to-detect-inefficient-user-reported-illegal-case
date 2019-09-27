import xgboost as xgb 
import pandas as pd 
import numpy as np
from sklearn import svm
from xgboost import XGBClassifier
import matplotlib.pylab as plt 
from matplotlib.pylab import rcParams
from sklearn.model_selection import cross_validate, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score


dfee0 = pd.read_csv("dfee0new.csv")

df = dfee0

target = 'status'
IDcol = 'uid'
predictors = [x for x in df.columns if x not in [target, IDcol]]


param_test2 = {
        'max_depth': range(3,10,1),
        'min_child_weight': range(1,6,2)
}

gsearch2 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.1, n_estimators = 2900,
                                                    max_depth = 4, min_child_weight = 1,
                                                    objective = 'binary:logistic', nthread = 8, subsample = 0.8, 
                                                    colsample_bytree = 0.8, gamma = 0, reg_alpha = 0, reg_lambda = 1,
                                                    scale_pos_weight = 1
                                                    ),
                            param_grid = param_test2,
                            scoring = 'roc_auc', 
                            n_jobs = 35,
                            cv = 5
                            )
gsearch2.fit(df[predictors], df['status'])
print(gsearch2.best_params_, gsearch2.best_score_)
