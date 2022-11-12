import pandas as pd
import numpy as np
import os as os
from pandas import DataFrame
import pickle
import json as json
import copy as copy
from orderedset import OrderedSet

# Constants
low_percentile = 25
middle_percentile = 50
high_percentile = 75
evaluation_latest_date_wrt_target = \
    [pd.Timedelta('2 days'), pd.Timedelta('7 days'), pd.Timedelta('14 days'), pd.Timedelta('21 days'), \
        pd.Timedelta('30 days'), pd.Timedelta('60 days'), pd.Timedelta('90 days')]
target_columns = ['ret_wrt_ind_fwd2d_log', 'ret_wrt_ind_fwd7d_log', 'ret_wrt_ind_fwd14d_log', 'ret_wrt_ind_fwd21d_log',
    'ret_wrt_ind_fwd30d_log', 'ret_wrt_ind_fwd60d_log', 'ret_wrt_ind_fwd90d_log']
target_names = ['2d', '7d', '14d', '21d', '30d', '60d', '90d']

def data_splitting(feature_data_file, feature_detail_file, saving_folder, best_stocks_file, \
    stocks_config_file, starting_date=None, test_split_proportion = 0.3, \
    eval_stocks_file=None, horizon=None, level=None, saving_prefix=None):
    # stocks config
    df_config = pd.read_csv(stocks_config_file)

    # best stocks
    if level is not None:
        best_stocks = list(df_config.iloc[np.where(df_config['level'] == level)[0]]['symbol'])
        eval_stocks = list(df_config.iloc[np.where(df_config['level'] == level)[0]]['symbol'])
    else:
        with open(best_stocks_file) as json_file:
            best_stocks = json.load(json_file)['stocks']

        # extra stocks
        if eval_stocks_file is not None:
            with open(eval_stocks_file) as json_file:
                eval_stocks = json.load(json_file)['stocks']

    # data read
    df_input = pd.read_csv(feature_data_file)
    df_input = df_input.replace([np.inf, -np.inf], np.nan)
    df = df_input.iloc[np.where([x in best_stocks for x in df_input['stock_name']])[0]]

    df_evstock = df_input.iloc[np.where([x in eval_stocks for x in df_input['stock_name']])[0]]

    with open(feature_detail_file) as json_file:
        feature_detail = json.load(json_file)

    # time dropping
    df['date'] = pd.to_datetime(df['date'])
    df_evstock['date'] = pd.to_datetime(df_evstock['date'])

    if starting_date is not None:
        df = df[df['date'] >= starting_date]
        df_evstock = df_evstock[df_evstock['date'] >= starting_date]

    # non-casual columns
    noncasual_features = list()
    for f,d in feature_detail.items():
        if 'casuality' in d:
            if d['casuality'] is False:
                noncasual_features.append(f)

    # feature columns
    feature_names = list((OrderedSet(list(df.columns)) - OrderedSet(noncasual_features)) - \
        OrderedSet(['Unnamed: 0', 'date', 'jdate', 'stock_name']))

    # horizon specification
    if horizon is not None:
        horizon_ind = target_names.index(horizon)
        target_names_list = [target_names[horizon_ind]]
        target_columns_list = [target_columns[horizon_ind]]
        evaluation_latest_date_wrt_target_list = [evaluation_latest_date_wrt_target[horizon_ind]]
    else:
        target_names_list = target_names
        target_columns_list = target_columns
        evaluation_latest_date_wrt_target_list = evaluation_latest_date_wrt_target

    # 
    result_dict = dict()
    result_dict['feature_names'] = feature_names
    result_dict['collections'] = dict()
    for trg_name, trg_col, evaluation_date in zip(target_names_list, target_columns_list, evaluation_latest_date_wrt_target_list):
        print(trg_name)

        ending_date = max(df['date'])
        evaluation_date = ending_date - evaluation_date
        df_new = copy.copy(df)
        df_evstock_new = copy.copy(df_evstock)

        # evaluation data 
        df_eval = df_new[df_new['date'] > evaluation_date]
        df_no_eval = df_new[df_new['date'] <= evaluation_date]

        df_evstock_new_new_eval = df_evstock_new[df_evstock_new['date'] > evaluation_date]
        df_eval = df_evstock_new_new_eval # if using the eval stock for evaluation

        # train test splitting
        random_indice = np.random.permutation(np.arange(df_no_eval.shape[0]))
        random_indice_test = random_indice[0:int(len(random_indice)*test_split_proportion),]
        random_indice_train = random_indice[int(len(random_indice)*test_split_proportion):,]

        df_train = df_no_eval.iloc[random_indice_train,]
        df_test = df_no_eval.iloc[random_indice_test,]

        # levels
        Llow = np.percentile(df_train[~df_train[trg_col].isnull()][trg_col], low_percentile)
        Lmid = np.percentile(df_train[~df_train[trg_col].isnull()][trg_col], middle_percentile)
        Lhigh = np.percentile(df_train[~df_train[trg_col].isnull()][trg_col], high_percentile)

        # labels
        df_new['labels'] = (df_new[trg_col] > Llow).astype(int) + \
            (df_new[trg_col] > Lmid).astype(int) + (df_new[trg_col] > Lhigh).astype(int)
        df_new['labels'] = df_new['labels'].mask(df_new[trg_col].isnull() == True)

        # indice
        train_indice = list(df_train.index)
        test_indice = list(df_test.index)
        evaluation_indice = list(df_eval.index)

        # target up to now
        def target_up_to_now_map(stock_group):
            initial_index = stock_group.index
            df = stock_group
            df.index = [pd.Timestamp(item) for item in df['date']]
            
            ind_return_up_to_now = (np.log(df['totalindex'][-1]) - np.log(df['totalindex'])).values
            stock_return_up_to_now = (np.log(df['close_price'][-1]) - np.log(df['close_price'])).values
            return_df = pd.DataFrame({\
                'target_up_to_now':stock_return_up_to_now - 0.5*ind_return_up_to_now, \
                })
            return_df.index = initial_index
            return return_df

        stock_groups = df_eval.groupby('stock_name')
        evaluation_target_up_to_now = stock_groups.apply(target_up_to_now_map).reset_index(drop=True)


        # saving folder creation
        if (os.path.isdir(saving_folder) is False):
            os.makedirs(saving_folder)

        # saving results
        result_dict['collections'][trg_name] = {
            'train_indice':train_indice, 
            'test_indice':test_indice, 
            'evaluation_indice':evaluation_indice, 
            'train_labels':list(df_new.loc[train_indice]['labels']), 
            'test_labels':list(df_new.loc[test_indice]['labels']), 
            'evaluation_target_up_to_now':list(evaluation_target_up_to_now['target_up_to_now'].values.reshape([-1])), 
            'Llow':Llow, 
            'Lmid':Lmid, 
            'Lhigh':Lhigh, 
        }

    # saving data
    if saving_prefix is None:
        saving_prefix = ''

    with open(saving_folder + '/' + saving_prefix + '_' + 'TestTrainSplitted.json', 'w') as outfile:
        json.dump(result_dict, outfile, indent=2, ensure_ascii=False)