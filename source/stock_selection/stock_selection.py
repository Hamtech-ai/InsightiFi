## library
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy.matlib
import operator
import numpy as np
from pandas import DataFrame
from numpy import array
from sklearn.metrics import accuracy_score
import json
import os 
from sklearn.metrics import classification_report
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from matplotlib import pyplot
import numpy.matlib
import sys
from sklearn.model_selection import train_test_split
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.pipeline import Pipeline                 
from mlxtend.feature_selection import ColumnSelector
from sklearn.metrics import roc_auc_score


def create_test_bench(Data, test_train_indices, model_horizon,regime):

    # indices are valid for raw data which contains nan

    test_train_horizon_data = test_train_indices['collections'][model_horizon]
    train_indice = test_train_horizon_data['train_indice']
    test_indice = test_train_horizon_data['test_indice']
    train_labels = test_train_horizon_data['train_labels']
    test_labels = test_train_horizon_data['test_labels']
    TrainLabel = np.array(train_labels)
    TestLabel = np.array(test_labels)
    Label  = np.concatenate([TrainLabel,TestLabel],axis=0)
    evaluation_indice = test_train_horizon_data['evaluation_indice']
    evaluation_labels = test_train_horizon_data['evaluation_labels']
    EvaluationData = Data.loc[evaluation_indice]
    EvaluationLabel = np.array(evaluation_labels)

    if (regime == 'Buy') :
        Data = pd.concat([Data.iloc[train_indice], Data.iloc[test_indice]], axis=0).reset_index(drop=True)
        test_indices = np.where((Data['date'] > '2019-03-21') & \
                                (Data['date'] < '2020-07-22') )
        TestData = Data.iloc[test_indices]
        TestLabel = Label[test_indices]
        train_indices = np.where((Data['date'] < '2018-12-21') | \
                                (Data['date'] > '2020-10-22') )
        TrainData = Data.iloc[train_indices]
        TrainLabel = Label[train_indices]

    elif (regime == 'Side' ):
        Data = pd.concat([Data.iloc[train_indice], Data.iloc[test_indice]], axis=0).reset_index(drop=True)
        test_indices = np.where((Data['date'] > '2016-04-13') & \
                                (Data['date'] < '2017-11-05') )
        TestData = Data.iloc[test_indices]
        TestLabel = Label[test_indices]
        train_indices = np.where((Data['date'] < '2016-01-13') | \
                                (Data['date'] > '2018-02-05') )
        TrainData = Data.iloc[train_indices]
        TrainLabel = Label[train_indices]

    elif( regime == 'Sell') :
        Data = pd.concat([Data.iloc[train_indice], Data.iloc[test_indice]], axis=0).reset_index(drop=True)
        test_indices = np.where((Data['date'] > '2014-02-01') & \
                                 (Data['date'] < '2016-01-09')  )
        TestData = Data.iloc[test_indices]
        TestLabel = Label[test_indices]
        train_indices = np.where((Data['date'] < '2013-11-01') | \
                                (Data['date'] > '2016-04-09') )
        TrainData = Data.iloc[train_indices]
        TrainLabel = Label[train_indices]

    elif (regime == 'Pivot') :
        Data = pd.concat([Data.iloc[train_indice], Data.iloc[test_indice]], axis=0).reset_index(drop=True)
        test_indices = np.where((Data['date'] > '2020-06-07') & \
                                 (Data['date'] < '2020-11-18')  )
        TestData = Data.iloc[test_indices]
        TestLabel = Label[test_indices]
        train_indices = np.where((Data['date'] < '2020-03-07') )
        TrainData = Data.iloc[train_indices]
        TrainLabel = Label[train_indices]

    elif (regime == 'Current'):
        train_indice = test_train_horizon_data['train_indice']
        test_indice = test_train_horizon_data['test_indice']
        train_labels = test_train_horizon_data['train_labels']
        test_labels = test_train_horizon_data['test_labels']
        TrainData = Data.loc[train_indice]
        TestData = Data.loc[test_indice]
        TrainLabel = np.array(train_labels)
        TestLabel = np.array(test_labels)

    else :
        print('invalid regime')
    return TrainData, TrainLabel, TestData, TestLabel, EvaluationData, EvaluationLabel

def anova_scores(TrainData, TrainLabel
                   ):
    # configure to select all features
    fs = SelectKBest(score_func=f_classif, k='all')
    # learn relationship from training data
    fs.fit(TrainData, TrainLabel)
    # transform train input data
    X_train_fs = fs.transform(TrainData)
    
    return X_train_fs,fs

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

def MI_scores(TrainData, TrainLabel,
                    ):
    # configure to select all features
    fs = SelectKBest(score_func=mutual_info_classif, k='all')
    # learn relationship from training data
    fs.fit(TrainData, TrainLabel)
    # transform train input data
    X_train_fs = fs.transform(TrainData)
    
    return X_train_fs, fs

def stock_selection_final(test_train_file, model_horizon, signal, stock_selection_params_file, regime = 'Buy'):

    model_description = 'StockSelection for '+ signal + ' Signal ' + model_horizon + ' Horizon'
    print('---------------------',model_description,'---------------------')
    # input path based on signal
    try:
        feature_data_file = 'data'+'/feature_data'  + '/' + signal +'_selected_feature_data' + '.csv'
    except: 
        feature_data_file = 'data'+'/feature_data'  + '/' + 'Buy' +'_selected_feature_data' + '.csv'
    # read dara
    feature_df = pd.read_csv(feature_data_file).drop(['Unnamed: 0'], axis=1)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

    # test train split read
    with open(test_train_file) as json_file:
        test_train_indices = json.load(json_file)

    # stock selection params read
    with open(stock_selection_params_file) as json_file:
        stock_selection_params = json.load(json_file)

    Data = feature_df
    
    #label info and splitting indices
    label_map = stock_selection_params['label_info'][signal]['label_map']
    TrainData_raw, TrainLabel_raw, TestData_raw, TestLabel_raw, EvaluationData_raw, EvaluationLabel_raw = create_test_bench(Data,\
         test_train_indices, model_horizon,regime)
    

    # nan indices
    train_is_not_nan = np.logical_not(np.logical_or(TrainData_raw.isnull().any(axis=1), np.isnan(TrainLabel_raw)))
    test_is_not_nan = np.logical_not(np.logical_or(TestData_raw.isnull().any(axis=1), np.isnan(TestLabel_raw)))
    evaluation_is_not_nan = np.logical_not(EvaluationData_raw.isnull().any(axis=1))

    
    train_allowable_indice = np.where(train_is_not_nan)[0]
    test_allowable_indice = np.where(test_is_not_nan)[0]
    evaluation_allowable_indice = np.where(evaluation_is_not_nan)[0]

    # nan-removing
    TrainData = TrainData_raw.iloc[train_allowable_indice]
    TestData = TestData_raw.iloc[test_allowable_indice]
    EvaluationData = EvaluationData_raw.iloc[evaluation_allowable_indice]

    TrainLabel = TrainLabel_raw[train_allowable_indice]
    TestLabel = TestLabel_raw[test_allowable_indice]
    EvaluationLabel = EvaluationLabel_raw[evaluation_allowable_indice]

    # punishments
    ## delete the buy_queue_locked = true
    TrainData_Punishments = np.where((TrainData['buy_queue_locked']!=1) & (TrainData['sell_queue_locked']!=1))[0]
    TestData_Punishments = np.where((TestData['buy_queue_locked']!=1) & (TestData['sell_queue_locked']!=1))[0]
    Evaluation_Punishments = np.where((EvaluationData['buy_queue_locked']!=1) & (EvaluationData['sell_queue_locked']!=1))[0]
    TrainData = TrainData.iloc[TrainData_Punishments]
    TestData = TestData.iloc[TestData_Punishments]
    EvaluationData = EvaluationData.iloc[Evaluation_Punishments]
    TrainLabel = TrainLabel[TrainData_Punishments]
    TestLabel = TestLabel[TestData_Punishments]
    EvaluationLabel = EvaluationLabel[Evaluation_Punishments]

    # get ready 
    TrainData = TrainData.drop(['buy_queue_locked','sell_queue_locked'],axis=1).reset_index(drop=True)
    TestData = TestData.drop(['buy_queue_locked','sell_queue_locked'],axis=1).reset_index(drop=True)
    EvaluationData = EvaluationData.drop(['buy_queue_locked','sell_queue_locked'],axis=1).reset_index(drop=True)



        # label transformation
    def label_transform(labels, label_map):
        indices_0 = np.where(labels == 0)[0]
        indices_1 = np.where(labels == 1)[0]
        indices_2 = np.where(labels == 2)[0]
        indices_3 = np.where(labels == 3)[0]
        
        final_labels = labels
        final_labels[indices_0] = label_map[0]
        final_labels[indices_1] = label_map[1]
        final_labels[indices_2] = label_map[2]
        final_labels[indices_3] = label_map[3]
        return final_labels

    TrainLabel_binary = label_transform(TrainLabel, label_map)
    TestLabel_binary = label_transform(TestLabel, label_map)

    ## saving date and stock name for results
    sel = ['date', 'stock_name']
    EvaluationProp = EvaluationData[sel]
    TrainData = TrainData.drop(sel,axis=1)
    TestData = TestData.drop(sel,axis=1)
    EvaluationData = EvaluationData.drop(sel,axis=1)

    ## saving date and stock name for results
    today = dt.datetime.today().strftime('%Y%m%d')  
    saving_folder = 'data'+'/Predictions' + '/Ensemble_' + signal + '_Signal' + today +'.csv'

    ## constants
    feature_num = (TrainData.shape[1])
    feature_names = list(TrainData.columns) ## names of the features

    #Normalizing Data
    scaler = RobustScaler()
    scaler.fit(TrainData)
    NormalizedTrainFeatures = scaler.transform(TrainData)
    NormalizedTestFeatures = scaler.transform(TestData)
    NormalizedEvaluationFeatures = scaler.transform(EvaluationData)


    # anova f-test feature selection for numerical data in RF
    X_train_fs,fs= anova_scores(NormalizedTrainFeatures, TrainLabel)

    # what are scores for the features
    for i in range(len(fs.scores_)):
        print('%s %d : %f' % (feature_names[i],i, fs.scores_[i]))

    scores = fs.scores_
    scores = np.array(scores)
    # selected columns
    rf_high_scores = np.where( scores >= 600)

    # MI test feature selection for numerical data in RF
    X_train_fs, fs = MI_scores(NormalizedTrainFeatures, TrainLabel)

    # what are scores for the features
    for i in range(len(fs.scores_)):
        print('%s %d : %f' % (feature_names[i],i, fs.scores_[i]))

    scores = fs.scores_
    scores = np.array(scores)
    MI_high_scores = np.where( scores >= 0.0065)

    # feed forward feature selection
    y = np.transpose(TrainLabel) ## dim =  1*data
    x = np.transpose(NormalizedTrainFeatures)  ## dim = features*trials
    F_new = 38
    f_num = (NormalizedTrainFeatures.shape[1])
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
    # Models
    params = stock_selection_params['rf_params'][signal][model_horizon]

    model_1 = RandomForestClassifier(n_estimators=params['n_estimators'][0], criterion=params['criterion'][0],\
                                bootstrap= params['bootstrap'][0],min_samples_leaf =params['min_samples_leaf'][0],\
                                    max_depth=params['max_depth'][0])
    model_2 = RandomForestClassifier(n_estimators=params['n_estimators'][1], criterion=params['criterion'][1],\
                                bootstrap= params['bootstrap'][1],min_samples_leaf=params['min_samples_leaf'][1],\
                                    max_depth=params['max_depth'][1])
    model_3 = RandomForestClassifier(n_estimators=params['n_estimators'][2], criterion=params['criterion'][2],\
                                bootstrap= params['bootstrap'][2],min_samples_leaf=params['min_samples_leaf'][2],\
                                    max_depth=params['max_depth'][2])     

    # ensemble, column selectors and pipelines

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
    print(dt.datetime.now())
    eclf = EnsembleVoteClassifier(clfs=[clf1_pipe,clf2_pipe,clf3_pipe],voting='soft')

    eclf.fit(NormalizedTrainFeatures,TrainLabel)
    print('done4/4')
    ensemble_test = eclf.predict(NormalizedTestFeatures)
    ensemble_prediction_prob = eclf.predict_proba(NormalizedEvaluationFeatures)
    ensemble_prediction = eclf.predict(NormalizedEvaluationFeatures)

    cm = confusion_matrix(y_target=TestLabel, y_predicted=ensemble_test ,binary=False)
    print('AUC =',roc_auc_score(TestLabel, ensemble_test, average=None))
    print('-----','confusion matrix for stock selection ', signal, ' signal in', regime, 'regime', '-----')
    print(classification_report(TestLabel, ensemble_test, labels=[0, 1]))
    EvaluationProp['Label'] = ensemble_prediction
    EvaluationProp['Score'] = ensemble_prediction_prob[:,1]

    EvaluationProp.to_csv(saving_folder)

    return














