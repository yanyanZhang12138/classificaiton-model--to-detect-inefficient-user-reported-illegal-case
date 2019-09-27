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
 
########################### modelfit function definition
#%%
def modelfit(alg, dtrain, predictors, performCV = True, printFeatureImportance = True, cv_folds = 5):
    #if useTrainCV:
        #xgb_param = alg.get_xgb_params()
        #xgtrain = xgb.DMatrix(dtrain[predictors].values, label = dtrain[target].values)
        #cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round = alg.get_params()['n_estimators'], nfold= cv_folds,
                       # metrics = 'auc', early_stopping_rounds = early_stopping_rounds, show_progress = False)
       # alg.set_params(n_estimators= cvresult.shape[0])

#fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['status'],eval_metric = 'auc')

#predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

#perform cross-validation
    if performCV:
        cv_score = cross_val_score(alg, dtrain[predictors], dtrain['status'], cv = cv_folds, n_jobs = -1, scoring = 'roc_auc')

#print model report
    print("\nModel Report")
    print("Accuracy : %.4g" % accuracy_score(dtrain['status'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % roc_auc_score(dtrain['status'], dtrain_predprob))

    if performCV:
        print("CV Score: Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))
#print Feature Importance
    if printFeatureImportance:
        feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
        feat_imp.plot(kind = 'bar', title = 'Feature Importances')
        plt.ylabel('Feature Importance Score')


#%% choose all predictors except target & IDcol and contentcol
#%% choose all predictors except target & IDcol and contentcol
param_test1 = {'learning_rate' : [0.01, 0.05, 0.07, 0.1, 0.2]}
gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.1,
                                                n_estimators = 2900, 
                                                max_depth = 7,
                                                min_child_weight = 1, 
                                                subsample=0.9,
                                                colsample_bytree = 0.9,
                                                gamma = 0.1,
                                                reg_alpha = 2,
                                                reg_lambda = 0.1,
                                                nthread=8),
                        param_grid = param_test1,
                        scoring = 'roc_auc',
                        n_jobs = 35, cv = 5)
gsearch1.fit(df[predictors], df['status'])
print(gsearch1.best_params_, gsearch1.best_score_)
