from stock_selection.memory_features import model_prediction
import datetime as dt



# input folders
test_train_file = 'TestTrainData' + '/' + 'TestTrainSplitted' + '.json'
feature_data_file = 'data'+'/feature_data'  + '/' + 'daily_feature_data' + '.csv'
stock_selection_params_file = 'configs' + '/stock_selection_params' + '.json'


# path
model_horizon = '2d'
signal = 'Buy'

model_prediction(feature_data_file, test_train_file, stock_selection_params_file, model_horizon,\
     signal)

# path
model_horizon = '7d'
signal = 'Buy'

model_prediction(feature_data_file, test_train_file, stock_selection_params_file, model_horizon,\
     signal)

# path
model_horizon = '14d'
signal = 'Buy'

model_prediction(feature_data_file, test_train_file, stock_selection_params_file, model_horizon,\
     signal)

# path
model_horizon = '21d'
signal = 'Buy'

model_prediction(feature_data_file, test_train_file, stock_selection_params_file, model_horizon,\
     signal)
