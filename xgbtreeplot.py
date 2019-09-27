#%%
import xgboost as xgb
import pandas as pd 
from sklearn import svm
from xgboost import XGBClassifier
from xgboost import plot_tree
import joblib
import graphviz
import numpy as np 
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.pylab import rcParams

model = joblib.load('rp_xgbmodel3.model')
dfee0 = pd.read_csv("dfee0new.csv")
dfee0['status'] = dfee0['status'].astype(int)

#%%
def create_feature_map(features):
    with open('xgb.fmap','w') as outfile:
        i = 0
        for feat in features:
            outfile.write('{0}\t{1}\tq\n'.format(i,feat))
            i = i+1
        outfile.close()


#%%
create_feature_map(dfee0.columns)

fig, ax = plt.subplots()
fig.set_size_inches(150, 100)
xgb.plot_tree(model, ax = ax, fmap='xgb.fmap', num_trees = 0)
fig.savefig('xgbtreeplot.png')
