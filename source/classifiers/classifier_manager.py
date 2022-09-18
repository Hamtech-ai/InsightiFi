from configs.param_getter import classifier_parameters, all_horizons, horizon_classifiers, label_name_horizon_classifiers
from .RF_classifier import RF_classifier
from .AdaBoost_classifier import AdaBoost_classifier
from .classifier_visualization import precision_recal_report, conf_execute, conf_mat

import json as json
import numpy as np
import pandas as pd
import copy as copy
import os as os
from sklearn.preprocessing import StandardScaler

def classifier_preparation(feature_data_file, test_train_file, model_horizon, model_name):
    # classifier config
    classifier_info = classifier_parameters(model_horizon, model_name)
    selected_features = classifier_info["selected_features"]
    model_description = classifier_info["model_description"]
    model_param = classifier_info["model_param"]
    label_name = classifier_info["label_name"]
    label_map = classifier_info["label_map"]

    # feature deteil read
    with open(test_train_file) as json_file:
        test_train_data = json.load(json_file)

    test_train_horizon_data = test_train_data['collections'][model_horizon]

    train_indice = test_train_horizon_data['train_indice']
    test_indice = test_train_horizon_data['test_indice']
    evaluation_indice = test_train_horizon_data['evaluation_indice']

    evaluation_target_up_to_now = test_train_horizon_data['evaluation_target_up_to_now']

    train_labels = test_train_horizon_data['train_labels']
    test_labels = test_train_horizon_data['test_labels']

    Llow = test_train_horizon_data['Llow']
    Lmid = test_train_horizon_data['Lmid']
    Lhigh = test_train_horizon_data['Lhigh']

    # feature data read
    feature_df = pd.read_csv(feature_data_file)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

    TrainData_raw = feature_df[selected_features].loc[train_indice]
    TestData_raw = feature_df[selected_features].loc[test_indice]
    EvaluationData_raw = feature_df[selected_features].loc[evaluation_indice]
    EvaluationIdentityData_raw = feature_df[['date', 'stock_name']].loc[evaluation_indice]

    TrainLabel_raw = np.array(train_labels)
    TestLabel_raw = np.array(test_labels)

    # label transformation
    def label_transform(labels):
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

    TrainLabel_raw = label_transform(TrainLabel_raw)
    TestLabel_raw = label_transform(TestLabel_raw)

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

    # normalizing
    scaler = StandardScaler().fit(TrainData)

    NormalizedTrainData = scaler.transform(TrainData)
    NormalizedTestData = scaler.transform(TestData)
    NormalizedEvaluationData = scaler.transform(EvaluationData)

    # raw labels
    raw_train_num = len(train_indice)
    raw_test_num = len(test_indice)
    raw_evaluation_num = len(evaluation_indice)

    return NormalizedTrainData, TrainLabel, NormalizedTestData, TestLabel, NormalizedEvaluationData, \
        evaluation_target_up_to_now, EvaluationIdentityData_raw, \
        train_allowable_indice, test_allowable_indice, evaluation_allowable_indice, \
        TrainLabel_raw, TestLabel_raw, raw_train_num, raw_test_num, raw_evaluation_num, \
        selected_features, model_description, label_name, model_param, \
        Llow, Lmid, Lhigh

def ensemble_preparation(result_folder, feature_data_file, test_train_file, model_horizon, ensemble_name, result_data_prefix):
    # classifier config
    classifier_info = classifier_parameters(model_horizon, ensemble_name)
    model_description = classifier_info["model_description"]
    model_param = classifier_info["model_param"]
    label_name = classifier_info["label_name"]
    label_map = classifier_info["label_map"]

    # collecting data
    weak_model_list = horizon_classifiers(model_horizon, just_model=True)
    result_data_files = [result_data_prefix + '_' + item + '.json' \
        for item in weak_model_list]

    train_decision_group = list()
    test_decision_group = list()
    evaluation_decision_group = list()

    train_decision_proba_group = list()
    test_decision_proba_group = list()
    evaluation_decision_proba_group = list()

    train_true_target = list()
    test_true_target = list()

    model_name_group = list()
    model_description_group = list()
    for file in result_data_files:
        print(file)
        with open(result_folder + '/' + file, 'rb') as json_file:
            res_dict = json.load(json_file)
            
        train_decision_group.append(res_dict['train_pred'])
        test_decision_group.append(res_dict['test_pred'])
        evaluation_decision_group.append(res_dict['evaluation_pred'])
        
        train_decision_proba_group.append(res_dict['train_pred_proba'])
        test_decision_proba_group.append(res_dict['test_pred_proba'])
        evaluation_decision_proba_group.append(res_dict['evaluation_pred_proba'])
        
        train_true_target.append(res_dict['train_true_target'])
        test_true_target.append(res_dict['test_true_target'])
        
        model_name_group.append(res_dict['model_name'])
        model_description_group.append(res_dict['model_description'])

    # feature deteil read
    with open(test_train_file) as json_file:
        test_train_data = json.load(json_file)

    test_train_horizon_data = test_train_data['collections'][model_horizon]

    evaluation_target_up_to_now = test_train_horizon_data['evaluation_target_up_to_now']

    train_indice = test_train_horizon_data['train_indice']
    test_indice = test_train_horizon_data['test_indice']
    evaluation_indice = test_train_horizon_data['evaluation_indice']

    train_labels = test_train_horizon_data['train_labels']
    test_labels = test_train_horizon_data['test_labels']

    TrainLabel_raw = np.array(train_labels)
    TestLabel_raw = np.array(test_labels)

    # feature data read
    feature_df = pd.read_csv(feature_data_file)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

    EvaluationIdentityData_raw = feature_df[['date', 'stock_name']].loc[evaluation_indice]

    # label transformation
    def label_transform(labels):
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

    TrainLabel_raw = label_transform(TrainLabel_raw)
    TestLabel_raw = label_transform(TestLabel_raw)

    # nan-removing
    # is not nan mask
    train_is_not_nan = np.logical_not(np.isnan(TrainLabel_raw))
    test_is_not_nan = np.logical_not(np.isnan(TestLabel_raw))
    evaluation_is_not_nan = np.array([True]*len(evaluation_indice))

    # allowable indice
    train_allowable_indice = np.where(train_is_not_nan)[0]
    test_allowable_indice = np.where(test_is_not_nan)[0]
    evaluation_allowable_indice = np.where(evaluation_is_not_nan)[0]

    # nan-removing
    ensemble_train_data = np.concatenate(train_decision_proba_group, axis=1)[train_allowable_indice, :]
    ensemble_test_data = np.concatenate(test_decision_proba_group, axis=1)[test_allowable_indice, :]
    ensemble_evaluation_data = np.concatenate(evaluation_decision_proba_group, axis=1)[evaluation_allowable_indice, :]

    ensemble_train_true_target = np.array(TrainLabel_raw)[train_allowable_indice].astype(int)
    ensemble_test_true_target = np.array(TestLabel_raw)[test_allowable_indice].astype(int)

    # raw labels
    raw_train_num = len(train_indice)
    raw_test_num = len(test_indice)
    raw_evaluation_num = len(evaluation_indice)

    return train_decision_group, test_decision_group, evaluation_decision_group, \
        evaluation_target_up_to_now, EvaluationIdentityData_raw, \
        train_decision_proba_group, test_decision_proba_group, evaluation_decision_proba_group, \
        ensemble_train_data, ensemble_test_data, ensemble_evaluation_data, \
        ensemble_train_true_target, ensemble_test_true_target, \
        train_allowable_indice, test_allowable_indice, evaluation_allowable_indice, \
        TrainLabel_raw, TestLabel_raw, raw_train_num, raw_test_num, raw_evaluation_num, \
        model_name_group, model_description, label_name, model_param

def best_model_preparation(result_folder, model_horizon, model_list, result_prefix):
    # collecting data
    result_data_files = [result_prefix + '_' + item + '.json' \
        for item in model_list]

    train_decision_group = list()
    test_decision_group = list()
    evaluation_decision_group = list()

    train_decision_proba_group = list()
    test_decision_proba_group = list()
    evaluation_decision_proba_group = list()

    train_true_target_group = list()
    test_true_target_group = list()

    model_name_group = list()
    model_description_group = list()
    for file in result_data_files:
        print(file)
        with open(result_folder + '/' + file, 'rb') as json_file:
            res_dict = json.load(json_file)
            
        train_decision_group.append(res_dict['train_pred'])
        test_decision_group.append(res_dict['test_pred'])
        evaluation_decision_group.append(res_dict['evaluation_pred'])
        
        train_decision_proba_group.append(res_dict['train_pred_proba'])
        test_decision_proba_group.append(res_dict['test_pred_proba'])
        evaluation_decision_proba_group.append(res_dict['evaluation_pred_proba'])
        
        train_true_target_group.append(res_dict['train_true_target'])
        test_true_target_group.append(res_dict['test_true_target'])
        
        model_name_group.append(res_dict['model_name'])
        model_description_group.append(res_dict['model_description'])

    return train_decision_group, test_decision_group, \
        train_true_target_group, test_true_target_group, model_name_group

def finalized_model_preparation(best_model_info, result_folder, result_prefix):
    # collecting data
    labels_df = dict()
    for label,item in best_model_info.items():
        labels_df[label] = pd.read_csv(result_folder + '/' + result_prefix + '_eval' + '_' + item['best_model_name'] + '.csv')

    return labels_df

def classifier_result_save(saving_folder, saving_prefix, model_horizon, model_name, \
    train_pred, test_pred, evaluation_pred, train_pred_proba, \
    test_pred_proba, evaluation_pred_proba, feature_importance, \
    evaluation_target_up_to_now, EvaluationIdentityData_raw, train_allowable_indice, test_allowable_indice, evaluation_allowable_indice, \
    TrainLabel_raw, TestLabel_raw, raw_train_num, raw_test_num, raw_evaluation_num, \
    selected_features, model_description, label_name):
    # Adding Nan to results
    final_train_pred = np.array([np.nan]*raw_train_num)
    final_test_pred = np.array([np.nan]*raw_test_num)
    final_evaluation_pred = np.array([np.nan]*raw_evaluation_num)

    final_train_pred[train_allowable_indice] = train_pred
    final_test_pred[test_allowable_indice] = test_pred
    final_evaluation_pred[evaluation_allowable_indice] = evaluation_pred

    # Adding 0 probability to results
    final_train_pred_proba = np.zeros([raw_train_num, train_pred_proba.shape[1]])
    final_test_pred_proba = np.zeros([raw_test_num, test_pred_proba.shape[1]])
    final_evaluation_pred_proba = np.zeros([raw_evaluation_num, evaluation_pred_proba.shape[1]])

    final_train_pred_proba[train_allowable_indice,] = train_pred_proba
    final_test_pred_proba[test_allowable_indice,] = test_pred_proba
    final_evaluation_pred_proba[evaluation_allowable_indice,] = evaluation_pred_proba

    final_train_true_target = TrainLabel_raw
    final_test_true_target = TestLabel_raw

    # decision certainity
    train_mask = np.eye(final_train_pred_proba.shape[1], dtype=int)[np.nan_to_num(final_train_pred, nan=0).astype(int)]
    test_mask = np.eye(final_test_pred_proba.shape[1], dtype=int)[np.nan_to_num(final_test_pred, nan=0).astype(int)]
    evaluation_mask = np.eye(final_evaluation_pred_proba.shape[1], dtype=int)[np.nan_to_num(final_evaluation_pred, nan=0).astype(int)]

    train_decision_certainity = np.sum(final_train_pred_proba*train_mask, axis=1)
    test_decision_certainity = np.sum(final_test_pred_proba*test_mask, axis=1)
    evaluation_decision_certainity = np.sum(final_evaluation_pred_proba*evaluation_mask, axis=1)

    # saving folder creation
    if (os.path.isdir(saving_folder) is False):
        os.makedirs(saving_folder)

    # saving the data
    result_dict = {
        'train_pred':list(final_train_pred), 
        'test_pred':list(final_test_pred), 
        'evaluation_pred':list(final_evaluation_pred), 
        'train_pred_proba':list(final_train_pred_proba.tolist()), 
        'test_pred_proba':list(final_test_pred_proba.tolist()), 
        'evaluation_pred_proba':list(final_evaluation_pred_proba.tolist()), 
        'train_true_target':list(final_train_true_target.tolist()), 
        'test_true_target':list(final_test_true_target.tolist()),
        'train_decision_certainity':list(train_decision_certainity), 
        'test_decision_certainity':list(test_decision_certainity), 
        'evaluation_decision_certainity':list(evaluation_decision_certainity), 
        'feature_importance':list(feature_importance.tolist()), 
        'model_name':model_name,
        'model_description':model_description, 
        'feature_names':selected_features, 
        'model_horizon':model_horizon, 
        'label_name':label_name
        }
    
    with open(saving_folder + '/' + saving_prefix + '_' + model_name + '.json', 'w') as outfile:
        json.dump(result_dict, outfile, indent=2, ensure_ascii=False)
    
    # saving evaluation results
    AugEvaluationData = copy.copy(EvaluationIdentityData_raw)
    AugEvaluationData["decision"] = final_evaluation_pred
    AugEvaluationData["certainity"] = evaluation_decision_certainity
    AugEvaluationData[list(range(final_evaluation_pred_proba.shape[1]))] = final_evaluation_pred_proba
    AugEvaluationData['target_up_to_now'] = evaluation_target_up_to_now
    AugEvaluationData.to_csv(saving_folder + '/' + saving_prefix + '_eval' + '_' + model_name + '.csv')

def single_classifier(feature_data_file, test_train_file, model_horizon, model_name, \
    saving_folder, model_saving_folder, saving_prefix, plot_conf_mat=False, precision_conf=False, eval_mode=False):
    # preparation
    TrainData, TrainLabel, TestData, TestLabel, EvaluationData, \
        evaluation_target_up_to_now, EvaluationIdentityData_raw, \
        train_allowable_indice, test_allowable_indice, evaluation_allowable_indice, \
        TrainLabel_raw, TestLabel_raw, raw_train_num, raw_test_num, raw_evaluation_num, \
        selected_features, model_description, label_name, model_param, \
        Llow, Lmid, Lhigh = \
            classifier_preparation(feature_data_file, test_train_file, model_horizon, model_name)

    # classiser
    # RF classifier
    train_pred, test_pred, evaluation_pred, train_pred_proba, \
        test_pred_proba, evaluation_pred_proba, feature_importance = \
            RF_classifier(TrainData, TrainLabel, TestData, EvaluationData, \
                params=model_param, model_saving_folder=model_saving_folder, \
                saving_prefix=saving_prefix, model_name=model_name, eval_mode=eval_mode)

    # result save
    classifier_result_save(saving_folder, saving_prefix, model_horizon, model_name, \
        train_pred, test_pred, evaluation_pred, train_pred_proba, \
        test_pred_proba, evaluation_pred_proba, feature_importance, \
        evaluation_target_up_to_now, EvaluationIdentityData_raw, train_allowable_indice, test_allowable_indice, evaluation_allowable_indice, \
        TrainLabel_raw, TestLabel_raw, raw_train_num, raw_test_num, raw_evaluation_num, \
        selected_features, model_description, label_name)
    
    # reporting
    print("Train Report")
    precision_recal_report(TrainLabel, train_pred)
    print("Test Report")
    precision_recal_report(TestLabel, test_pred)

    # confusion matrix
    if plot_conf_mat is True:
        conf_mat(TrainLabel, train_pred, name="Train", precision_conf=precision_conf)
        conf_mat(TestLabel, test_pred, name="Test", precision_conf=precision_conf)

def single_ensemble_classifier(feature_data_file, test_train_file, model_horizon, ensemble_name, \
    classifier_result_prefix, saving_folder, model_saving_folder, saving_prefix, \
    plot_conf_mat=False, precision_conf=False, eval_mode=False):
    # preparation
    train_decision_group, test_decision_group, evaluation_decision_group, \
        evaluation_target_up_to_now, EvaluationIdentityData_raw, \
        train_decision_proba_group, test_decision_proba_group, evaluation_decision_proba_group, \
        ensemble_train_data, ensemble_test_data, ensemble_evaluation_data, \
        ensemble_train_true_target, ensemble_test_true_target, \
        train_allowable_indice, test_allowable_indice, evaluation_allowable_indice, \
        TrainLabel_raw, TestLabel_raw, raw_train_num, raw_test_num, raw_evaluation_num, \
        model_name_group, model_description, label_name, model_param = \
            ensemble_preparation(saving_folder, feature_data_file, test_train_file, model_horizon, ensemble_name, classifier_result_prefix)

    # classiser
    # AdaBoost classifier
    train_pred, test_pred, evaluation_pred, train_pred_proba, \
        test_pred_proba, evaluation_pred_proba, feature_importance = \
            AdaBoost_classifier(ensemble_train_data, ensemble_train_true_target, ensemble_test_data, ensemble_evaluation_data, \
                params=model_param, model_saving_folder=model_saving_folder, saving_prefix=saving_prefix, model_name=ensemble_name, eval_mode=eval_mode)

    # preparation
    classifier_result_save(saving_folder, saving_prefix, model_horizon, ensemble_name, \
        train_pred, test_pred, evaluation_pred, train_pred_proba, \
        test_pred_proba, evaluation_pred_proba, feature_importance, \
        evaluation_target_up_to_now, EvaluationIdentityData_raw, train_allowable_indice, test_allowable_indice, evaluation_allowable_indice, \
        TrainLabel_raw, TestLabel_raw, raw_train_num, raw_test_num, raw_evaluation_num, \
        model_name_group, model_description, label_name)

    # reporting
    print("Train Report")
    precision_recal_report(ensemble_train_true_target, train_pred)
    print("Test Report")
    precision_recal_report(ensemble_test_true_target, test_pred)

    # confusion matrix
    if plot_conf_mat is True:
        conf_mat(ensemble_train_true_target, train_pred, name="Train", precision_conf=precision_conf)
        conf_mat(ensemble_test_true_target, test_pred, name="Test", precision_conf=precision_conf)

def label_name_best_model(feature_data_file, test_train_file, model_horizon, \
    model_list, result_folder, result_prefix, plot_conf_mat=False, precision_conf=False, \
    alpha_recal=None, label_weight=None):
    # preparation
    train_decision_group, test_decision_group, \
        train_true_target_group, test_true_target_group, model_name_group = \
            best_model_preparation(result_folder, model_horizon, model_list, result_prefix)

    # scoring the models
    score_group = list()
    percision_group = list()
    recal_group = list()
    for true_label,pred_label in zip(test_decision_group, test_true_target_group):
        conf_matrix = conf_execute(true_label,pred_label)
        label_precision = np.diag(conf_matrix)/np.sum(conf_matrix, axis=1)
        label_recal = np.diag(conf_matrix)/np.sum(conf_matrix, axis=0)

        print('precision = ' + str(label_precision) + ', recal = ' + str(label_recal))

        if (alpha_recal is not None) or (label_weight is not None):
            score_group.append(np.sum(
                np.array(label_weight)*(label_precision + label_recal*np.array(alpha_recal))))
        else:
            score_group.append(np.sum(label_precision) + np.sum(label_recal))
        
        percision_group.append(label_precision)
        recal_group.append(label_recal)

    # best model
    best_model_info = dict()
    best_ind = np.argmax(score_group)
    best_model_info['best_model_name'] = model_name_group[best_ind]
    best_model_info['precision'] = list(percision_group[best_ind])
    best_model_info['recal'] = list(recal_group[best_ind])

    best_train_pred = train_decision_group[best_ind]
    best_test_pred = test_decision_group[best_ind]
    best_train_true = train_true_target_group[best_ind]
    best_test_true = test_true_target_group[best_ind]

    # best declare
    print('*** Best is ' + best_model_info['best_model_name'])

    # reporting
    print("Train Report")
    precision_recal_report(best_train_true, best_train_pred)
    print("Test Report")
    precision_recal_report(best_test_true, best_test_pred)

    # confusion matrix
    if plot_conf_mat is True:
        conf_mat(best_train_true, best_train_pred, name="Train", precision_conf=precision_conf)
        conf_mat(best_test_true, best_test_pred, name="Test", precision_conf=precision_conf)
    
    return best_model_info

def horizon_execute(feature_data_file, test_train_file, saving_folder, model_saving_folder, model_horizon, \
    saving_prefix=None, eval_mode=False):
    classifier_list = horizon_classifiers(model_horizon, just_model=True)
    ensemble_list = horizon_classifiers(model_horizon, just_ensemble=True)
    if saving_prefix is None:
        saving_prefix = ''

    # execute classifiers
    model_saving_prefix = saving_prefix + '_' + 'result'
    for model in classifier_list:
        print('>>>>>>>>>> ' + model + ' <<<<<<<<<<')
        single_classifier(feature_data_file, test_train_file, model_horizon, model, \
            saving_folder, model_saving_folder, model_saving_prefix, eval_mode=eval_mode)

    # execute ensemble
    classifier_result_prefix = saving_prefix + '_' + 'result'
    saving_prefix = saving_prefix + '_' + 'result'
    for model in ensemble_list:
        print('>>>>>>>>>> ' + model + ' <<<<<<<<<<')
        single_ensemble_classifier(feature_data_file, test_train_file, model_horizon, model, \
            classifier_result_prefix, saving_folder, model_saving_folder, saving_prefix, eval_mode=eval_mode)

def horizon_best_model_find(feature_data_file, test_train_file, result_folder, model_saving_folder, model_horizon, \
    plot_conf_mat=False, precision_conf=False, saving_prefix=None):
    label_name_classifiers, labels_info = label_name_horizon_classifiers(model_horizon)
    
    if saving_prefix is None:
        saving_prefix = ''
    
    # label best for horizon models
    result_dict = {}
    result_prefix = saving_prefix + '_' + 'result'
    for label_name,model_list in label_name_classifiers.items():
        print('----- B E S T -----> ' + label_name + ' <----- B E S T -----')
        # label info
        label_info = labels_info[label_name]

        best_model_info = label_name_best_model(feature_data_file, test_train_file, model_horizon, \
            model_list, result_folder, result_prefix, plot_conf_mat=plot_conf_mat, precision_conf=precision_conf, \
            alpha_recal=label_info["alpha_recal"], label_weight=label_info["label_weight"])
        
        result_dict[label_name] = best_model_info

    # saving data
    saving_output_prefix = saving_prefix + '_' + 'best_result'
    with open(model_saving_folder + '/' + saving_output_prefix + '_' + model_horizon + '.json', 'w') as outfile:
        json.dump(result_dict, outfile, indent=2, ensure_ascii=False)
  
def eval_horizon_finalize(feature_data_file, test_train_file, result_folder, model_saving_folder, model_horizon, saving_prefix=None):
    print('---------- FINAL DECISION ----------')

    if saving_prefix is None:
        saving_prefix = ''

    best_model_prefix = saving_prefix + '_' + 'best_result'
    with open(model_saving_folder + '/' + best_model_prefix + '_' + model_horizon + '.json', 'rb') as infile:
        best_model_info = json.load(infile)

    label_name_classifiers, labels_info = label_name_horizon_classifiers(model_horizon)

    result_prefix = saving_prefix + '_' + 'result'
    labels_df = finalized_model_preparation(best_model_info, result_folder, result_prefix)

    # label best for horizon models
    buy_decision = labels_df["Buy"]["decision"]
    sell_decision = labels_df["Sell"]["decision"]
    sign_decision = labels_df["Sign"]["decision"]

    buy_certainity = labels_df["Buy"]["certainity"]
    sell_certainity = labels_df["Sell"]["certainity"]
    sign_certainity = labels_df["Sign"]["certainity"]
    
    # final score
    final_score = np.zeros(len(sign_certainity))

    # final buy
    final_buy_decision = np.zeros(len(buy_decision))
    buy_condition = np.logical_and(\
        np.logical_and(buy_decision == 1, sign_decision != 0), sell_decision != 1)
    buy_indice = np.where(buy_condition)[0]
    final_buy_decision[buy_indice] = 1
    final_score[buy_indice] = 2 + buy_certainity[buy_indice]

    # final sell
    final_sell_decision = np.zeros(len(sell_decision))
    sell_condition = np.logical_and(\
        np.logical_and(sell_decision == 1, sign_decision != 1), buy_decision != 1)
    sell_indice = np.where(sell_condition)[0]
    final_sell_decision[sell_indice] = 1
    final_score[sell_indice] = - 2 - sell_certainity[sell_indice]

    # final positive only
    final_positive_only_decision = np.zeros(len(buy_decision))
    positive_only_condition = np.logical_and(\
        np.logical_and(sign_decision == 1, buy_decision != 1), sell_decision != 1)
    positive_only_indice = np.where(positive_only_condition)[0]
    final_positive_only_decision[positive_only_indice] = 1
    final_score[positive_only_indice] = 1 + sign_certainity[positive_only_indice]

    # final negative only
    final_negative_only_decision = np.zeros(len(buy_decision))
    negative_only_condition = np.logical_and(\
        np.logical_and(sign_decision == 0, sell_decision != 1), buy_decision != 1)
    negative_only_indice = np.where(negative_only_condition)[0]
    final_negative_only_decision[negative_only_indice] = 1
    final_score[negative_only_indice] = - 1 - sign_certainity[negative_only_indice]

    # final not sure
    final_not_sure_decision = np.zeros(len(buy_decision))
    not_sure_condition = np.logical_not(np.logical_or(\
        np.logical_or(buy_condition, sell_condition), \
        np.logical_or(positive_only_condition, negative_only_condition)))
    not_sure_indice = np.where(not_sure_condition)[0]
    final_not_sure_decision[not_sure_indice] = 1
    final_score[not_sure_indice] = (\
        np.nan_to_num(2*sign_decision[not_sure_indice] - 1, nan=0)*sign_certainity[not_sure_indice] + \
        + np.nan_to_num(buy_decision[not_sure_indice], nan=0)*buy_certainity[not_sure_indice] \
        - np.nan_to_num(sell_decision[not_sure_indice], nan=0)*sell_certainity[not_sure_indice] \
        )/2

    # evaluation results
    AugEvaluationData = labels_df["Buy"][["date", "stock_name"]]
    AugEvaluationData["buy"] = final_buy_decision
    AugEvaluationData["sell"] = final_sell_decision
    AugEvaluationData["positive_only"] = final_positive_only_decision
    AugEvaluationData["negative_only"] = final_negative_only_decision
    AugEvaluationData["not_sure"] = final_not_sure_decision
    AugEvaluationData["score"] = final_score
    AugEvaluationData["target_up_to_now"] = labels_df["Buy"]["target_up_to_now"]

    # saving data
    if saving_prefix is None:
        saving_prefix = ''
    AugEvaluationData.to_csv(result_folder + '/' + saving_prefix + '_' + 'final_eval_' + model_horizon + '.csv')
    
def horizon_execute_all_process(feature_data_file, test_train_file, saving_folder, model_saving_folder, model_horizon, \
    plot_conf_mat=False, precision_conf=False, saving_prefix=None, eval_mode=False):
    horizon_execute(feature_data_file, test_train_file, saving_folder, model_saving_folder, model_horizon, \
        saving_prefix=saving_prefix, eval_mode=eval_mode)
    if eval_mode is False:
        horizon_best_model_find(feature_data_file, test_train_file, saving_folder, model_saving_folder, model_horizon, \
            plot_conf_mat=plot_conf_mat, precision_conf=precision_conf, saving_prefix=saving_prefix)
    eval_horizon_finalize(feature_data_file, test_train_file, saving_folder, model_saving_folder, model_horizon, saving_prefix=saving_prefix)

def all_execute(feature_data_file, test_train_file, saving_folder, model_saving_folder, \
    horizon=None, saving_prefix=None, eval_mode=False):
    # horizon specification
    if horizon is not None:
        horizons = [horizon]
    else:
        horizons = all_horizons()
    
    # all horizon running
    for model_horizon in horizons:
        print('########## ' + model_horizon + ' ##########')
        horizon_execute_all_process(feature_data_file, test_train_file, saving_folder, model_saving_folder, model_horizon, \
            plot_conf_mat=False, precision_conf=False, saving_prefix=saving_prefix, eval_mode=eval_mode)