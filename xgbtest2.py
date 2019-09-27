import xgboost as xgb 
import pandas as pd 
import numpy as np
from numpy import sort
from sklearn import svm
from xgboost import XGBClassifier
from xgboost import plot_importance
import matplotlib.pyplot as plt
plt.switch_backend('agg') 
from matplotlib.pylab import rcParams
from sklearn.model_selection import cross_validate, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
import joblib
import graphviz

dfee0 = pd.read_csv("dfee0new.csv")
df = dfee0
dfee0['status'] = dfee0['status'].astype(int)

target = 'status'
IDcol = 'uid'
predictors = [x for x in df.columns if x not in [target, IDcol]]
x_train, x_test, y_train, y_test = train_test_split(dfee0[predictors], dfee0[target], test_size = 0.2)

def modelfit(alg, train_data_features, train_labels, useTrainCV = True, cv_folds = 5):
    if useTrainCV:
        params = alg.get_xgb_params()
        xgb_param = dict([(key, [params[key]]) for key in params])
        
        boost = xgb.XGBClassifier()
        cvresult = GridSearchCV(boost, xgb_param, cv = cv_folds)
        cvresult.fit(train_data_features, train_labels)
        alg = cvresult.best_estimator_

#fit the algorithm on the data
    alg.fit(train_data_features, train_labels)

#predict training set:
    dtrain_predictions = alg.predict(train_data_features)
    dtrain_predprob = alg.predict_proba(train_data_features)[:,1]

#perform cross-validation
   # if performCV:
    #   cv_score = cross_val_score(alg, dtrain[predictors], dtrain['status'], cv = cv_folds, n_jobs = -1, scoring = 'roc_auc')

#print model report
    print("\nModel Report")
    print("Accuracy : %.4g" % accuracy_score(train_labels, dtrain_predictions))
    print("AUC Score (Train): %f" % roc_auc_score(train_labels, dtrain_predprob))

   # if performCV:
   #     print("CV Score: Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))


dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

xgb2 = XGBClassifier(learning_rate = 0.05, n_estimators = 2900,
                    max_depth = 7, min_child_weight = 1, gamma = 0.1,
                    subsample = 0.9, colsample_bytree = 0.9,
                    reg_alpha = 2, reg_lambda = 0.1, 
                    objective = 'binary:logistic', nthread = 8,
                    scale_pos_weight = 1)
modelfit(xgb2, np.array(x_train), np.array(y_train))

params = {
    'learning_rate' : 0.05,
    'booster' : 'gbtree',
    'num_class' : 2,
    'n_estimator' : 2900,
    'max_depth' : 7,
    'min_child_weight' : 1,
    'gamma' : 0.1,
    'subsample' : 0.9,
    'colsample_bytree' : 0.9,
    'reg_alpha' : 2,
    'reg_lambda' : 0.1,
    'objective' : 'multi:softmax',
    'nthread' : 8,
    'scale_pos_weight' : 1
    }

plst = params.items()
     
num_rounds = 500

model = xgb.train(plst, dtrain, num_rounds)

ans = model.predict(dtest)

y_test1 = np.asarray(y_test)


cm1 = pd.crosstab(y_test, ans, rownames = ['Actual'], colnames = ['Predicted'])
print(cm1)

plt.figure(num = 1, figsize = (12,8))
plot_importance(model)
plt.show()
plt.savefig('xgbmodel3.png')

#fit model on all training data
xgb2.fit(x_train, y_train)
ypred = xgb2.predict(x_test)
predictions = [round(value) for value in ypred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
#fit model using each importance as a threshold
thresholds = sort(xgb2.feature_importances_)
for thresh in thresholds:
    #select feature using thresholds
    selection = SelectFromModel(xgb2, threshold=thresh, prefit = True)
    select_x_train = selection.transform(x_train)
    #train mode;
    selection_model = XGBClassifier(learning_rate = 0.05, n_estimators = 2900,
                                     max_depth = 7, min_child_weight = 1, gamma = 0.1,
                                     subsample = 0.9, colsample_bytree = 0.9,
                                     reg_alpha = 2, reg_lambda = 0.1, 
                                     objective = 'binary:logistic', nthread = 8,
                                     scale_pos_weight = 1)
    selection_model.fit(select_x_train, y_train)
    #eval model
    select_x_test = selection.transform(x_test)
    ypred2 = selection_model.predict(select_x_test)
    predictions = [round(value) for value in ypred2]
    accuracy = accuracy_score(y_test, predictions)
    print("Thresh = %.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_x_train.shape[1], accuracy*100.0))


joblib.dump(model,'rp_xgbmodel3.model')


