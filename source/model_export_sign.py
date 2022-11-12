from stock_selection.model_export import saved_models_prediction
import pandas as pd
import numpy as np
import json

# signal param
signal = 'Sign'

# data inputs
test_train_file = 'TestTrainData' + '/' + 'TestTrainSplitted' + '.json'
feature_data_file = 'data'+'/feature_data'  + '/' + 'daily_feature_data' + '.csv'
saving_folder_file = 'data'+'/feature_data'  + '/' + signal + '_selected_feature_data' + '.csv'
stock_selection_params_file = 'configs' + '/stock_selection_params' + '.json'

# load all features for final merge
feature_df = pd.read_csv(feature_data_file)
with open(stock_selection_params_file) as json_file:
        stock_selection_params = json.load(json_file)
initial_selected_features = stock_selection_params['feature_info']
feature_df = feature_df[initial_selected_features]


## 2d horizon
model_horizon = '2d'
EvaluationProp_2d = saved_models_prediction(test_train_file, feature_df, model_horizon, signal, stock_selection_params_file )
feature_df = feature_df.merge(EvaluationProp_2d, on=['date','stock_name'], how = 'left')

## 7d horizon
model_horizon = '7d'
EvaluationProp_7d = saved_models_prediction(test_train_file, feature_df, model_horizon, signal, stock_selection_params_file )
feature_df = feature_df.merge(EvaluationProp_7d, on=['date','stock_name'], how = 'left')

## 14d horizon
model_horizon = '14d'
EvaluationProp_14d = saved_models_prediction(test_train_file, feature_df, model_horizon, signal, stock_selection_params_file )
feature_df = feature_df.merge(EvaluationProp_14d, on=['date','stock_name'], how = 'left')


model_horizon = '21d'
EvaluationProp_21d = saved_models_prediction(test_train_file, feature_df, model_horizon, signal, stock_selection_params_file )
feature_df = feature_df.merge(EvaluationProp_21d, on=['date','stock_name'], how = 'left')




feature_df.to_csv(saving_folder_file)