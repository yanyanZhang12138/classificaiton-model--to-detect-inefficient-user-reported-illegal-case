#!/usr/bin/python3
#-*- coding:utf-8 -*-
import pandas as pd 
import numpy as np
from imblearn.ensemble import EasyEnsemble

df=pd.read_table('reportupdate3.txt', delim_whitespace = True, names=('uid', 'people_count', 'user_level', 'fans_count', 'official', 'money', 'live_count', 'diamond', 'content_cnt', 'mon', 'status', 'punish_rate', 'judge'))
del df['judge']
del df['official']
del df['mon']
print(df.head())

#print(df[(df['diamond'] == '2019-02-12')].index.tolist())
def data_processing():
    df['people_count'] = df['people_count'].astype(int)
    df["user_level"] = df["user_level"].astype(int)
    df['fans_count'] = df['fans_count'].astype(int)
    df['money'] = df['money'].astype(int)
    df['live_count'] = df['live_count'].astype(int)
    df['diamond'] = df['diamond'].astype(int)
    df['uid']=df['uid'].astype(object)
    df['content_cnt'] = df['content_cnt'].astype(int)
    df['punish_rate'] = df['punish_rate'].replace('\\N', '0')
    df['punish_rate'] = df['punish_rate'].astype(float)
    df['status']=df['status'].replace('1','0')
    df['status']=df['status'].replace('2','0')
    df['status']=df['status'].replace('3','0')
    df['status']=df['status'].replace('4','0')
    df['status']=df['status'].replace('5','0')
    df['status']=df['status'].replace('6','1')
    df['status']=df['status'].replace('101','1')
    df['status']=df['status'].replace('13','0')
    df['status']=df['status'].replace('\\N', '1')
    df['status']=df['status'].replace('N', '1')
    



data_processing()


df_train = df.loc[:, ['status','people_count', 'user_level', 'fans_count', 'money', 'live_count', 'diamond', 'content_cnt', 'punish_rate']]
print(df_train.head())
#################################### detect and replace outliers

def replaceOutliers(x, df):
    median = x.median()
    std = x.std()
    outliers = (x - median).abs() > std
    x[outliers] = np.nan
    x.fillna(median, inplace = True)
    return x


replaceOutliers(df_train['people_count'], df_train)
replaceOutliers(df_train['user_level'], df_train)
replaceOutliers(df_train['fans_count'], df_train)
replaceOutliers(df_train['money'], df_train)
replaceOutliers(df_train['live_count'], df_train)
replaceOutliers(df_train['diamond'], df_train)
replaceOutliers(df_train['content_cnt'], df_train)
replaceOutliers(df_train['punish_rate'], df_train)

##########################################seperate x, y and store them to a different data frame 

x_train = df_train.loc[:, ['people_count', 'user_level', 'fans_count', 'money', 'live_count', 'diamond', 'content_cnt', 'punish_rate']]
y_train = df_train.loc[:, ['status']]
print(x_train.head())
print(y_train.head())


from imblearn.ensemble import EasyEnsemble
ee = EasyEnsemble(random_state = 0, n_subsets = 10, replacement=True)
x_ee, y_ee = ee.fit_sample(x_train, y_train)
print(x_ee.shape)

dfee0 = pd.DataFrame(x_ee[0],columns=['people_count', 'user_level', 'fans_count', 'money', 'live_count', 'diamond', 'content_cnt', 'punish_rate'])
dfee0['status'] = y_ee[0]

dfee1 = pd.DataFrame(x_ee[1],columns=['people_count', 'user_level', 'fans_count', 'money', 'live_count', 'diamond', 'content_cnt', 'punish_rate'])
dfee1['status'] = y_ee[1]

dfee2 = pd.DataFrame(x_ee[2],columns=['people_count', 'user_level', 'fans_count', 'money', 'live_count', 'diamond', 'content_cnt', 'punish_rate'])
dfee2['status'] = y_ee[2]

dfee3 = pd.DataFrame(x_ee[3],columns=['people_count', 'user_level', 'fans_count', 'money', 'live_count', 'diamond', 'content_cnt', 'punish_rate'])
dfee3['status'] = y_ee[3]

dfee4 = pd.DataFrame(x_ee[4],columns=['people_count', 'user_level', 'fans_count', 'money', 'live_count', 'diamond', 'content_cnt', 'punish_rate'])
dfee4['status'] = y_ee[4]

dfee5 = pd.DataFrame(x_ee[5],columns=['people_count', 'user_level', 'fans_count', 'money', 'live_count', 'diamond', 'content_cnt', 'punish_rate'])
dfee5['status'] = y_ee[5]

dfee6 = pd.DataFrame(x_ee[6],columns=['people_count', 'user_level', 'fans_count', 'money', 'live_count', 'diamond', 'content_cnt', 'punish_rate'])
dfee6['status'] = y_ee[6]

dfee7 = pd.DataFrame(x_ee[7],columns=['people_count', 'user_level', 'fans_count', 'money', 'live_count', 'diamond', 'content_cnt', 'punish_rate'])
dfee7['status'] = y_ee[7]

dfee8 = pd.DataFrame(x_ee[8],columns=['people_count', 'user_level', 'fans_count', 'money', 'live_count', 'diamond', 'content_cnt', 'punish_rate'])
dfee8['status'] = y_ee[8]

dfee9 = pd.DataFrame(x_ee[9],columns=['people_count', 'user_level', 'fans_count', 'money', 'live_count', 'diamond', 'content_cnt', 'punish_rate'])
dfee9['status'] = y_ee[9]

dfee0.to_csv('dfee0new.csv',index = False)
dfee1.to_csv('dfee1new.csv',index = False)
dfee2.to_csv('dfee2new.csv',index = False)
dfee3.to_csv('dfee3new.csv',index = False)
dfee4.to_csv('dfee4new.csv',index = False)
dfee5.to_csv('dfee5new.csv',index = False)
dfee6.to_csv('dfee6new.csv',index = False)
dfee7.to_csv('dfee7new.csv',index = False)
dfee8.to_csv('dfee8new.csv',index = False)
dfee9.to_csv('dfee9new.csv',index = False)
