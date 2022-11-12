# import library
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import random
import json
from sklearn.metrics import classification_report
import copy 

import datetime
# data export

model_horizon = '30d'
feature_data = 'data'+'/feature_data' + '/overall_feature_data' + '.csv'
level_info = 'configs' + '/stock_ids.csv'
model_description = 'MarketTiming for 30-Day Horizon'
print('---------------------',model_description,'---------------------')

features = pd.read_csv(feature_data)

features = features.reset_index(drop=True)
features = features.drop(['Unnamed: 0'],axis=1)
features = features.replace([np.inf, -np.inf], np.nan)
features['date'] = features['date'].astype('datetime64[ns]')
ending_date = pd.Series([max(features['date'])])
duration = pd.Series([pd.Timedelta('30 days')])
evaluation_start_date = ending_date - duration
df_new = copy.copy(features)

# splitting
df_train = df_new[df_new['date'] < str(evaluation_start_date[0])]
df_test = df_new[df_new['date'] >= str(evaluation_start_date[0])]


# preprocessing
columns_to_drop = ['wlvl_ret_fwd30d_log','wlvl_nonindv_power3d','wlvl_nonindv_power5d',
'wlvl_nonindv_power7d',]
TrainData_raw = df_train.drop(columns_to_drop,axis=1)
TestData_raw = df_test.drop(columns_to_drop,axis=1)
TrainLabel_raw =df_train['wlvl_ret_fwd30d_log']
TrainLabel_raw = np.array(TrainLabel_raw)
TestLabel_raw = df_test['wlvl_ret_fwd30d_log']
TestLabel_raw = np.array(TestLabel_raw)

# nan-removing
# is not nan mask
train_is_not_nan = np.logical_not(np.logical_or(TrainData_raw.isnull().any(axis=1),
                                                np.isnan(TrainLabel_raw)))
test_is_not_nan = np.logical_not(TestData_raw.isnull().any(axis=1))

# allowable indice
train_allowable_indice = np.where(train_is_not_nan)[0]
test_allowable_indice = np.where(test_is_not_nan)[0]

# nan-removing
TrainData = TrainData_raw.iloc[train_allowable_indice]
TestData = TestData_raw.iloc[test_allowable_indice]

## no need for labels
TrainLabel_cont = TrainLabel_raw[train_allowable_indice]

# indentifying labels
L1 = np.percentile(TrainLabel_cont,25)
L2 = np.percentile(TrainLabel_cont,50)
L3 = np.percentile(TrainLabel_cont,75)

TrainLabel = [0 if (i <= L1) else 1 if (i <= L2) else 2 if \
               (i <= L3) else 3 for i in TrainLabel_cont]

TrainLabel = np.array(TrainLabel)


# TestTrain Split
def create_test_bench(Data,Label,regime = 'Buy'):
    if (regime == 'Buy') : 
        test_indices = np.where((Data['date'] > '2019-03-21') & \
                                (Data['date'] < '2020-07-22') )
        TestData = Data.iloc[test_indices]
        TestLabel = Label[test_indices]
        train_indices = np.where((Data['date'] < '2018-12-21') | \
                                (Data['date'] > '2020-10-22') )
        TrainData = Data.iloc[train_indices]
        TrainLabel = Label[train_indices]
    elif (regime == 'Side' ):
        test_indices = np.where((Data['date'] > '2016-04-13') & \
                                (Data['date'] < '2017-11-05') )
        TestData = Data.iloc[test_indices]
        TestLabel = Label[test_indices]
        train_indices = np.where((Data['date'] < '2016-01-13') | \
                                (Data['date'] > '2018-02-05') )
        TrainData = Data.iloc[train_indices]
        TrainLabel = Label[train_indices]
    
    elif( regime == 'Sell') : 
        test_indices = np.where((Data['date'] > '2014-02-01') & \
                                 (Data['date'] < '2016-01-09')  )
        TestData = Data.iloc[test_indices]
        TestLabel = Label[test_indices]
        train_indices = np.where((Data['date'] < '2013-11-01') | \
                                (Data['date'] > '2016-04-09') )
        TrainData = Data.iloc[train_indices]
        TrainLabel = Label[train_indices]
        
    elif (regime == 'Pivot') : 
        test_indices = np.where((Data['date'] > '2020-06-07') & \
                                 (Data['date'] < '2020-11-18')  )
        TestData = Data.iloc[test_indices]
        TestLabel = Label[test_indices]
        train_indices = np.where((Data['date'] < '2020-03-07') )
        TrainData = Data.iloc[train_indices]
        TrainLabel = Label[train_indices]
    else :
        print('invalid regime')
        
    return TrainData, TrainLabel, TestData, TestLabel


X_train, y_train, X_test, y_test = create_test_bench(TrainData,TrainLabel,\
                                                    regime = 'Pivot')

# Keep extreme labels for train
TrainLabel_sharp_indices = np.where((y_train==0 )| (y_train==3))
X_train_sharp = X_train.iloc[TrainLabel_sharp_indices]
y_train_sharp = y_train[TrainLabel_sharp_indices]


## to binary function
def to_binary(labels):
    indices_0 = np.where( labels>=3  )
#     indices_1 = np.where(( labels==1 )| (labels == 2))
    TrainLabel_binary = np.zeros(len(labels),dtype=np.int8)
    TrainLabel_binary[indices_0,] = 1
#     TrainLabel_binary[indices_1,] = 1
    return(TrainLabel_binary)

y_train_binary = to_binary(y_train_sharp)

# get ready 
drop_list = ['date','level','wlvl_value20d']
Eval_Prop = pd.DataFrame(TestData['date'],columns=['date'])
X_train = X_train_sharp.drop(drop_list,axis=1)
X_test = X_test.drop(drop_list,axis=1)
X_eval = TestData.drop(drop_list,axis=1)

today = datetime.datetime.today().strftime('%Y%m%d')  
saving_folder = 'data'+'/Predictions' +'/MarketTiming_'+ today + '.csv'

#Normalizing Data
from sklearn.preprocessing import RobustScaler
cols = X_train.columns
scaler = RobustScaler()
scaler.fit(X_train)
NormalizedTrainData = scaler.transform(X_train)
# NormalizedTrainData = pd.DataFrame(NormalizedTrainData,columns=cols)
NormalizedTestData = scaler.transform(X_test)
# NormalizedTestData = pd.DataFrame(NormalizedTestData,columns=cols)
NormalizedEvaluationData = scaler.transform(X_eval)


# feature selection
# anova f-test feature selection for numerical data in RF
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from matplotlib import pyplot

def select_features(TrainData, TrainLabel
                   ):
    # configure to select all features
    fs = SelectKBest(score_func=f_classif, k='all')
    # learn relationship from training data
    fs.fit(TrainData, TrainLabel)
    # transform train input data
    X_train_fs = fs.transform(TrainData)
    
    return X_train_fs, fs

# feature selection
X_train_fs, fs = select_features(NormalizedTrainData, y_train_binary)
# what are scores for the features
C = X_train.columns
for i in range(len(fs.scores_)):
    print('%s %d : %f' % (C[i],i, fs.scores_[i]))

scores = fs.scores_
scores = np.array(scores)
rf_high_scores = np.where( scores >= 200)

# MI f-test feature selection for numerical data in RF

def select_features(TrainData, TrainLabel
                   ):
    # configure to select all features
    fs = SelectKBest(score_func=mutual_info_classif, k='all')
    # learn relationship from training data
    fs.fit(TrainData, TrainLabel)
    # transform train input data
    X_train_fs = fs.transform(TrainData)
    
    return X_train_fs, fs

# feature selection
X_train_fs,fs = select_features(NormalizedTrainData, y_train_binary )
# what are scores for the features
C = X_train.columns
for i in range(len(fs.scores_)):
    print('%s %d : %f' % (C[i],i, fs.scores_[i]))

scores = fs.scores_
scores = np.array(scores)
MI_high_scores = np.where( scores >= 0.0065)

## fisher function
import numpy.matlib
import sys
def fisherfunction(x,y,numf):
    Tr = len(y)  ## number of trials
    class0 = np.where(y==0)
    class1 = np.where(y==1)
    p0 = len(class0[0])/Tr  
    p1 = len(class1[0])/Tr  
    x0 = x[:,class0] ; x0 = np.reshape(x0,(numf,len(class0[0])))  
    x1 = x[:,class1] ; x1 = np.reshape(x1,(numf,len(class1[0])))
    m0 = np.mean(x0,axis=1) 
    m1 = np.mean(x1,axis=1)
    M = p0*m0 + p1*m1
    res0 = np.transpose(np.matlib.repmat(m0, len(class0[0]),1))
    res1 = np.transpose(np.matlib.repmat(m1, len(class1[0]),1))
    S0 = np.matmul((x0-res0),np.transpose((x0-res0)))/len(class0[0])
    S1 = np.matmul((x1-res1),np.transpose((x1-res1)))/len(class1[0])
    Sw = p0*S0 + p1*S1
    m0 = np.reshape(m0,(numf,1))
    m1 = np.reshape(m1,(numf,1))
    M = np.reshape(M,(numf,1))
    Sb = p0*np.matmul(m0-M,np.transpose(m0-M)) + p1*np.matmul(m1-M,np.transpose(m1-M))
    J = np.trace(Sb)/np.trace(Sw)
    return(J)

# Feature selection
y = np.transpose(y_train_binary) ## dim =  1*data
x = np.transpose(np.array(NormalizedTrainData)) ## dim = features*trials
F_new = 50
f_num = (NormalizedTrainData.shape[1])
numf = 1  ### counter for features which are selected
selfeatures = []### cell for index of selected features
sel_flag = np.zeros([f_num,])  ### check if a feature is considered or not
while(numf < F_new):
    fishermat = np.zeros([1,f_num])
    
    for f in range(f_num):
        if (sel_flag[f]==0):
            Newfeatures = selfeatures.copy()
            Newfeatures.append(f)
            fishermat[:,f] = fisherfunction(x[Newfeatures,:],y,numf)
    
    maxl = np.argmax(fishermat)
    selfeatures.append(maxl)
    sel_flag[maxl] = 1
    numf = numf+1

## 
from mlxtend.feature_selection import ColumnSelector
from sklearn.model_selection import train_test_split
from mlxtend.evaluate import confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

model_1 = RandomForestClassifier(n_estimators=1000, criterion='entropy',\
                               bootstrap= True,class_weight='balanced', max_depth=20)
model_2 = RandomForestClassifier(n_estimators=1000, criterion='entropy',\
                               bootstrap= True,class_weight='balanced', max_depth=25)
model_3 = RandomForestClassifier(n_estimators=1000, criterion='entropy',\
                               bootstrap= True,class_weight='balanced', max_depth=30)


from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.pipeline import Pipeline                 
from mlxtend.feature_selection import ColumnSelector

col_sel1 = ColumnSelector(cols=list(rf_high_scores[0]))
col_sel2 = ColumnSelector(cols=list(MI_high_scores[0]))
col_sel3 = ColumnSelector(cols=list(selfeatures))

clf1_pipe = Pipeline([('sel', col_sel1),
                      ('RF1', model_1)])
print('done1/4')
clf2_pipe = Pipeline([('sel', col_sel2),
                      ('RF2', model_2)])
print('done2/4')
clf3_pipe = Pipeline([('sel', col_sel3),
                      ('RF3', model_3)])
print('done3/4')

eclf = EnsembleVoteClassifier(clfs=[clf1_pipe,clf2_pipe,clf3_pipe],voting='soft')
data = np.concatenate([NormalizedTrainData,NormalizedTestData],axis=0)
label = np.concatenate([y_train,y_test],axis=0)
model_1.fit(NormalizedTrainData,y_train_binary)

# plt.barh(C, model_1.feature_importances_)
# plt.barh(C, model_2.feature_importances_)
# plt.barh(C, model_3.feature_importances_)

print('done4/4')
ensemble_prediction = model_1.predict_proba(NormalizedEvaluationData)
ensemble_test = model_1.predict_proba(NormalizedTestData)
test_with_thr = ensemble_test>0.6
pred_with_thr = ensemble_prediction>0.6

# test data
label = np.zeros(len(ensemble_test),)
for i in range(len(ensemble_test)):
    if ( (test_with_thr[i,0]==False) & (test_with_thr[i,1]==True)):
        label[i] = 3 
    if ((test_with_thr[i,0]==False) & (test_with_thr[i,1]==False) & \
       (ensemble_test[i,0]<ensemble_test[i,1])):
        label[i] = 2
    if ((test_with_thr[i,0]==False) & (test_with_thr[i,1]==False)& \
       (ensemble_test[i,0]>ensemble_test[i,1])):
        label[i] = 1
    elif ((test_with_thr[i,0]==True) & (test_with_thr[i,1]==False)):
        label[i] = 0

label = np.array(label)

# cm = confusion_matrix(y_target=y_test, y_predicted=label ,binary=False)
# # plot test properties
# fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.title('=====Ensemble=====')
plt.show()
print(classification_report(y_test, label, labels=[0, 1, 2, 3]))

# evaluation data
label = np.zeros(len(ensemble_prediction),)
for i in range(len(ensemble_prediction)):
    if ( (pred_with_thr[i,0]==False) & (pred_with_thr[i,1]==True)):
        label[i] = 2
    if ((pred_with_thr[i,0]==False) & (pred_with_thr[i,1]==False)):
        label[i] = 1
    elif ((pred_with_thr[i,0]==True) & (pred_with_thr[i,1]==False)):
        label[i] = 0
        
label = np.array(label)

Eval_Prop['Label'] = label
Eval_Prop.to_csv(saving_folder)
