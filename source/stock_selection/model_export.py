import pickle
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

import datetime as dt


def saved_models_prediction(test_train_file, feature_df, model_horizon, signal, stock_selection_params_file ):

    print('----'+'export model for ' + signal + ' signal ' +model_horizon + ' horizon' +'----') 

    # memory models path
    mdl_filename = "saved_models/" + signal+ "_model_"+model_horizon+".pkl"

    # Export fitted models
    with open(mdl_filename, 'rb') as file:
        model = pickle.load(file)
        

        

    with open(test_train_file) as json_file:
        test_train_data = json.load(json_file)
    with open(stock_selection_params_file) as json_file:
        stock_selection_params = json.load(json_file)
    label_map = stock_selection_params['label_info'][signal]['label_map']
    feature_names_json = test_train_data['feature_names']
    test_train_horizon_data = test_train_data['collections'][model_horizon]

    train_indice = test_train_horizon_data['train_indice']
    test_indice = test_train_horizon_data['test_indice']
    evaluation_indice = test_train_horizon_data['evaluation_indice']

    train_labels = test_train_horizon_data['train_labels']
    test_labels = test_train_horizon_data['test_labels']
    evaluation_labels = test_train_horizon_data['evaluation_labels']


    # feature data read

    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

    # split

    TrainData_raw = feature_df.loc[train_indice]
    TestData_raw = feature_df.loc[test_indice]
    EvaluationData_raw = feature_df.loc[evaluation_indice].reset_index(drop=True)

    TrainLabel_raw = np.array(train_labels)
    TestLabel_raw = np.array(test_labels)
    EvaluationLabel_raw = np.array(evaluation_labels)
    
    # nan-removing
    TrainLabel_raw = np.array(train_labels)
    TestLabel_raw = np.array(test_labels)
    Data_raw = pd.concat([TrainData_raw,TestData_raw],axis=0).reset_index(drop=True)
    Label_raw = np.concatenate([TrainLabel_raw,TestLabel_raw],axis=0)

    data_isnan = np.where(Data_raw.isnull().any(axis=1) | np.isnan(Label_raw))
    eval_data_isnan = np.where((EvaluationData_raw.isnull().any(axis=1)))

    Data = Data_raw.drop(data_isnan[0],axis=0)
    Label = np.delete(Label_raw, data_isnan)
    EvaluationData = EvaluationData_raw.drop(eval_data_isnan[0],axis=0)

    # punishments
    ## delete the buy_queue_locked = true
    Data_Punishments = np.where((Data['buy_queue_locked']!=1) & (Data['sell_queue_locked']!=1))[0]
    Evaluation_Punishments = np.where((EvaluationData['buy_queue_locked']!=1) & (EvaluationData['sell_queue_locked']!=1))[0]
    Data = Data.iloc[Data_Punishments]
    EvaluationData = EvaluationData.iloc[Evaluation_Punishments]
    Label = Label[Data_Punishments]

    # get ready
    Data = Data.drop(['buy_queue_locked','sell_queue_locked'],axis=1).reset_index(drop=True)
    EvaluationData = EvaluationData.drop(['buy_queue_locked','sell_queue_locked'],axis=1).reset_index(drop=True)
    ## saving date and stock name for results
    all_data = pd.concat([Data, EvaluationData],axis=0)
    sel = ['date', 'stock_name']
    EvaluationProp = all_data[sel]
    Data = Data.drop(sel,axis=1)
    EvaluationData = EvaluationData.drop(sel,axis=1)

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

    Label_binary = label_transform(Label, label_map)

    print(dt.datetime.now())
        #Normalizing Data
    scaler = RobustScaler()
    scaler.fit(Data)
    NormalizedFeatures = scaler.transform(Data)
    NormalizedEvalFeatures = scaler.transform(EvaluationData)
    NormalizedEvalFeatures, EvaluationProp, Label_binary

    pred = model.predict(NormalizedEvalFeatures)
    new_feature = np.concatenate([Label_binary, pred],axis=0)
    EvaluationProp[signal+'_'+model_horizon] = new_feature

    return  EvaluationProp



