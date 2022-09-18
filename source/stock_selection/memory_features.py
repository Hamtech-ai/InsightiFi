# library
## library
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import classification_report
from sklearn.preprocessing import RobustScaler


def model_prediction(feature_data_file, test_train_file, stock_selection_params_file, model_horizon, \
                     signal):
    model_description = 'Stock ' + signal + ' for ' + model_horizon + ' Horizon'
    print('---------------------', model_description, '---------------------')

    # memory features path

    TwoDayHorizon = 'data' + '/PredictionFeatures' + '/' + signal + '2d_horizon.csv'
    SevenDayHorizon = 'data' + '/PredictionFeatures' + '/' + signal + '7d_horizon.csv'
    FourtnDayHorizon = 'data' + '/PredictionFeatures' + '/' + signal + '14d_horizon.csv'
    TwentyOneDayHorizon = 'data' + '/PredictionFeatures' + '/' + signal + '21d_horizon.csv'

    # memory models path
    mdl_filename_2d = "saved_models/" + signal + "_model_2d.pkl"
    mdl_filename_7d = "saved_models/" + signal + "_model_7d.pkl"
    mdl_filename_14d = "saved_models/" + signal + "_model_14d.pkl"
    mdl_filename_21d = "saved_models/" + signal + "_model_21d.pkl"

    with open(test_train_file) as json_file:
        test_train_data = json.load(json_file)
    with open(stock_selection_params_file) as json_file:
        stock_selection_params = json.load(json_file)
    initial_selected_features = stock_selection_params['feature_info']
    label_map = stock_selection_params['label_info'][signal]['label_map']
    feature_df = pd.read_csv(feature_data_file)
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
    feature_names_raw = initial_selected_features

    TrainData_raw = feature_df[initial_selected_features].loc[train_indice]
    TestData_raw = feature_df[initial_selected_features].loc[test_indice]
    EvaluationData_raw = feature_df[initial_selected_features].loc[evaluation_indice]

    TrainLabel_raw = np.array(train_labels)
    TestLabel_raw = np.array(test_labels)
    EvaluationLabel_raw = np.array(evaluation_labels)
    # nan-removing
    # is not nan mask
    train_is_not_nan = np.logical_not(np.logical_or(TrainData_raw.isnull().any(axis=1), np.isnan(TrainLabel_raw)))
    test_is_not_nan = np.logical_not(np.logical_or(TestData_raw.isnull().any(axis=1), np.isnan(TestLabel_raw)))
    evaluation_is_not_nan = np.logical_not(EvaluationData_raw.isnull().any(axis=1))

    # allowable indice
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
    TrainData_Punishments = np.where((TrainData['buy_queue_locked'] != 1) & (TrainData['sell_queue_locked'] != 1))[0]
    TestData_Punishments = np.where((TestData['buy_queue_locked'] != 1) & (TestData['sell_queue_locked'] != 1))[0]
    Evaluation_Punishments = \
    np.where((EvaluationData['buy_queue_locked'] != 1) & (EvaluationData['sell_queue_locked'] != 1))[0]
    TrainData = TrainData.iloc[TrainData_Punishments]
    TestData = TestData.iloc[TestData_Punishments]
    EvaluationData = EvaluationData.iloc[Evaluation_Punishments]
    TrainLabel = TrainLabel[TrainData_Punishments]
    TestLabel = TestLabel[TestData_Punishments]
    EvaluationLabel = EvaluationLabel[Evaluation_Punishments]

    if (model_horizon == '2d'):

        # get ready
        TrainData = TrainData.drop(['buy_queue_locked', 'sell_queue_locked'], axis=1).reset_index(drop=True)
        TestData = TestData.drop(['buy_queue_locked', 'sell_queue_locked'], axis=1).reset_index(drop=True)
        EvaluationData = EvaluationData.drop(['buy_queue_locked', 'sell_queue_locked'], axis=1).reset_index(drop=True)


    elif (model_horizon == '7d'):
        # get ready (adding two-day horizon feature)
        two_day_feature = pd.read_csv(TwoDayHorizon).reset_index(drop=True)
        two_day_feature = two_day_feature.drop(['Unnamed: 0'], axis=1)
        predicted_features = two_day_feature
        # merge and drops
        TrainData = TrainData.drop(['buy_queue_locked', 'sell_queue_locked'], axis=1).reset_index(drop=True)
        TestData = TestData.drop(['buy_queue_locked', 'sell_queue_locked'], axis=1).reset_index(drop=True)
        EvaluationData = EvaluationData.drop(['buy_queue_locked', 'sell_queue_locked'], axis=1).reset_index(drop=True)
        EvaluationData = EvaluationData.merge(predicted_features, on=['stock_name', 'date'], how='left')
        TrainData = TrainData.merge(predicted_features, on=['date', 'stock_name'], how='left').reset_index(drop=True)
        TestData = TestData.merge(predicted_features, on=['date', 'stock_name'], how='left').reset_index(drop=True)
        # drop probable nans from chaging horizons
        eval_nan = np.where(EvaluationData.isnull().any(axis=1))
        EvaluationData = EvaluationData.drop(eval_nan[0], axis=0).reset_index(drop=True)


    elif (model_horizon == '14d'):
        # get ready (adding two-day horizon feature)
        two_day_feature = pd.read_csv(TwoDayHorizon).reset_index(drop=True)
        two_day_feature = two_day_feature.drop(['Unnamed: 0'], axis=1)
        seven_day_feature = pd.read_csv(SevenDayHorizon).reset_index(drop=True)
        seven_day_feature = seven_day_feature.drop(['Unnamed: 0'], axis=1)
        predicted_features = two_day_feature.merge(seven_day_feature, on=['date', 'stock_name'], how='left')
        # merge
        TrainData = TrainData.drop(['buy_queue_locked', 'sell_queue_locked'], axis=1).reset_index(drop=True)
        TestData = TestData.drop(['buy_queue_locked', 'sell_queue_locked'], axis=1).reset_index(drop=True)
        EvaluationData = EvaluationData.drop(['buy_queue_locked', 'sell_queue_locked'], axis=1).reset_index(drop=True)
        EvaluationData = EvaluationData.merge(predicted_features, on=['stock_name', 'date'], how='left')
        TrainData = TrainData.merge(predicted_features, on=['date', 'stock_name'], how='left')
        TestData = TestData.merge(predicted_features, on=['date', 'stock_name'], how='left')

        # drop probable nans from chaging horizons
        eval_nan = np.where(EvaluationData.isnull().any(axis=1))
        EvaluationData = EvaluationData.drop(eval_nan[0], axis=0).reset_index(drop=True)

    else:
        # get ready (adding two-day horizon feature)
        two_day_feature = pd.read_csv(TwoDayHorizon).reset_index(drop=True)
        two_day_feature = two_day_feature.drop(['Unnamed: 0'], axis=1)
        seven_day_feature = pd.read_csv(SevenDayHorizon).reset_index(drop=True)
        seven_day_feature = seven_day_feature.drop(['Unnamed: 0'], axis=1)
        fourteen_day_feature = pd.read_csv(FourtnDayHorizon).reset_index(drop=True)
        fourteen_day_feature = fourteen_day_feature.drop(['Unnamed: 0'], axis=1)
        predicted_features_temp = two_day_feature.merge(seven_day_feature, on=['date', 'stock_name'], how='left')
        predicted_features = predicted_features_temp.merge(fourteen_day_feature, on=['date', 'stock_name'], how='left')
        # merge
        TrainData = TrainData.drop(['buy_queue_locked', 'sell_queue_locked'], axis=1).reset_index(drop=True)
        TestData = TestData.drop(['buy_queue_locked', 'sell_queue_locked'], axis=1).reset_index(drop=True)
        EvaluationData = EvaluationData.drop(['buy_queue_locked', 'sell_queue_locked'], axis=1).reset_index(drop=True)
        EvaluationData = EvaluationData.merge(predicted_features, on=['stock_name', 'date'], how='left')
        TrainData = TrainData.merge(predicted_features, on=['date', 'stock_name'], how='left')
        TestData = TestData.merge(predicted_features, on=['date', 'stock_name'], how='left')

        # drop probable nans from chaging horizons
        eval_nan = np.where(EvaluationData.isnull().any(axis=1))
        EvaluationData = EvaluationData.drop(eval_nan[0], axis=0).reset_index(drop=True)

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
    all_data = pd.concat([TrainData, TestData, EvaluationData], axis=0)
    sel = ['date', 'stock_name']
    EvaluationProp = all_data[sel]
    TrainData = TrainData.drop(sel, axis=1)
    TestData = TestData.drop(sel, axis=1)
    EvaluationData = EvaluationData.drop(sel, axis=1)

    ## constants
    feature_num = (TrainData.shape[1])
    feature_names = list(TrainData.columns)  ## names of the features

    # Normalizing Data

    scaler = RobustScaler()
    scaler.fit(TrainData)
    NormalizedTrainFeatures = scaler.transform(TrainData)
    NormalizedTestFeatures = scaler.transform(TestData)
    NormalizedEvaluationFeatures = scaler.transform(EvaluationData)

    # anova f-test feature selection for numerical data in RF

    def select_features(TrainData, TrainLabel_binary
                        ):
        # configure to select all features
        fs = SelectKBest(score_func=f_classif, k='all')
        # learn relationship from training data
        fs.fit(TrainData, TrainLabel_binary)
        # transform train input data
        X_train_fs = fs.transform(TrainData)

        return X_train_fs, fs  # X_test_fs, X_eval_fs, fs

    # feature selection
    X_train_fs, fs = select_features(NormalizedTrainFeatures, TrainLabel_binary)
    for i in range(len(fs.scores_)):
        print('%s %d : %f' % (feature_names[i], i, fs.scores_[i]))

    scores = fs.scores_
    scores = np.array(scores)

    # Models
    n_estimators = stock_selection_params['rf_params'][signal][model_horizon]['n_estimators'][0]

    model_1 = RandomForestClassifier(n_estimators=n_estimators, criterion='entropy', \
                                     bootstrap=True, max_depth=10, min_samples_leaf=1, \
                                     )

    model_1.fit(NormalizedTrainFeatures, TrainLabel_binary)
    print('done4/4')
    ensemble_test = model_1.predict(NormalizedTestFeatures)
    ensemble_prediction = model_1.predict(NormalizedEvaluationFeatures)

    print(classification_report(TestLabel, ensemble_test, labels=[0, 1]))

    a = np.concatenate([TrainLabel_binary, TestLabel_binary, ensemble_prediction], axis=0)

    if (model_horizon == '2d'):

        EvaluationProp[signal + '_' + model_horizon] = a
        EvaluationProp.to_csv(TwoDayHorizon)

        with open(mdl_filename_2d, 'wb') as file:
            pickle.dump(model_1, file)
    elif (model_horizon == '7d'):

        EvaluationProp[signal + '_' + model_horizon] = a
        EvaluationProp.to_csv(SevenDayHorizon)

        with open(mdl_filename_7d, 'wb') as file:
            pickle.dump(model_1, file)
    elif (model_horizon == '14d'):

        EvaluationProp[signal + '_' + model_horizon] = a
        EvaluationProp.to_csv(FourtnDayHorizon)

        with open(mdl_filename_14d, 'wb') as file:
            pickle.dump(model_1, file)

    elif (model_horizon == '21d'):

        EvaluationProp[signal + '_' + model_horizon] = a
        EvaluationProp.to_csv(TwentyOneDayHorizon)

        with open(mdl_filename_21d, 'wb') as file:
            pickle.dump(model_1, file)
    return
