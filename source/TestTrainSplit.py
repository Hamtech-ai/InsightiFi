import pandas as pd
import numpy as np
import os as os
from pandas import DataFrame
import pickle
import json as json
import copy as copy
from orderedset import OrderedSet

from utils.persian_utils import convert_to_fa_with_en_num

# Constants
test_latest_date = pd.Timedelta('365 days')
evaluation_latest_date_wrt_target = \
    [pd.Timedelta('2 days'), pd.Timedelta('7 days'), pd.Timedelta('14 days'), pd.Timedelta('21 days'), \
        pd.Timedelta('30 days'), pd.Timedelta('60 days'), pd.Timedelta('90 days')]
target_columns = ['ret_wrt_ind_fwd2d_log', 'ret_wrt_ind_fwd7d_log', 'ret_wrt_ind_fwd14d_log', 'ret_wrt_ind_fwd21d_log',
    'ret_wrt_ind_fwd30d_log', 'ret_wrt_ind_fwd60d_log', 'ret_wrt_ind_fwd90d_log']
target_names = ['2d', '7d', '14d', '21d', '30d', '60d', '90d']
saving_folder = 'TestTrainData'
feature_data_file = 'data/feature_data' + '/' + 'daily_feature_data' + '.csv'
feature_detail_file = 'data/feature_data' + '/' + 'daily_feature_detail' + '.json'
best_stocks_file = 'configs' + '/' + 'best_stocks' + '.json'
evaluation_stocks_file = 'configs' + '/' + 'eval_stocks' + '.json'
starting_date = pd.Timestamp('2010-04-01') # if None, the beggining of the time
#endind_date = pd.Timestamp('2020-10-13')
#evaluation_date = endind_date - pd.Timedelta('1 y')

print('---------------------TestTrainSplit---------------------')

# best stocks
with open(best_stocks_file) as json_file:
    best_stocks = json.load(json_file)['stocks']
best_stocks = [convert_to_fa_with_en_num(x) for x in best_stocks]
with open(evaluation_stocks_file) as json_file:
    evaluation_stocks = json.load(json_file)['stocks']
evaluation_stocks = [convert_to_fa_with_en_num(x) for x in evaluation_stocks]
# data read
df = pd.read_csv(feature_data_file)
df = df.replace([np.inf, -np.inf], np.nan)
df_product_test = df.iloc[np.where([x in evaluation_stocks for x in df['stock_name']])[0]]
df = df.iloc[np.where([x in best_stocks for x in df['stock_name']])[0]]

with open(feature_detail_file) as json_file:
    feature_detail = json.load(json_file)

# time dropping
df['date'] = pd.to_datetime(df['date'])
df_product_test['date'] = pd.to_datetime(df_product_test['date'])

if starting_date is not None:
    df = df[df['date'] >= starting_date]
    df_product_test = df_product_test[df_product_test['date'] >= starting_date]

# non-casual columns
noncasual_features = list()
for f,d in feature_detail.items():
    if 'casuality' in d:
        if d['casuality'] is False:
            noncasual_features.append(f)

# feature columns
feature_names = list((OrderedSet(list(df.columns)) - OrderedSet(noncasual_features)) - \
    OrderedSet(['Unnamed: 0', 'date', 'jdate', 'stock_name']))


#
result_dict = dict()
result_dict['feature_names'] = feature_names
result_dict['collections'] = dict()
for trg_name, trg_col, evaluation_date in zip(target_names, target_columns, evaluation_latest_date_wrt_target):
    print(trg_name)

    ending_date = max(df['date'])
    evaluation_start_date = ending_date - evaluation_date
    test_start_date = evaluation_start_date - test_latest_date
    df_new = copy.copy(df)
    df_new_product = copy.copy(df_product_test) 

    # splitting
    df_train = df_new[df_new['date'] < test_start_date]
    df_test = df_new[np.logical_and(df['date'] >= test_start_date, df_new['date'] < evaluation_start_date)]
    df_eval = df_new_product[df_new_product['date'] >= evaluation_start_date]

    # levels
    L15 = np.percentile(df_train[~df_train[trg_col].isnull()][trg_col], 25)
    L50 = np.percentile(df_train[~df_train[trg_col].isnull()][trg_col], 50)
    L85 = np.percentile(df_train[~df_train[trg_col].isnull()][trg_col], 75)

    # labels
    df_new['labels'] = (df_new[trg_col] > L15).astype(int) + \
        (df_new[trg_col] > L50).astype(int) + (df_new[trg_col] > L85).astype(int)
    df_new['labels'] = df_new['labels'] = df_new['labels'] = df_new['labels'] = df_new['labels'].mask(df_new[trg_col].isnull() == True)

    df_new_product['labels'] = (df_new_product[trg_col] > L15).astype(int) + \
        (df_new_product[trg_col] > L50).astype(int) + (df_new_product[trg_col] > L85).astype(int)
    df_new_product['labels'] = df_new_product['labels'] = df_new_product['labels'] = df_new_product['labels'] = df_new_product['labels'].mask(df_new_product[trg_col].isnull() == True)

    # indice
    train_indice = list(df_train.index)
    test_indice = list(df_test.index)
    evaluation_indice = list(df_eval.index)
    
    # saving results
    result_dict['collections'][trg_name] = {
        
        'train_indice':train_indice, 
        'test_indice':test_indice, 
        'evaluation_indice':evaluation_indice, 
        'train_labels':list(df_new.loc[train_indice]['labels']), 
        'test_labels':list(df_new.loc[test_indice]['labels']), 
        'evaluation_labels':list(df_new_product.loc[evaluation_indice]['labels']), 
        'L15':L15, 
        'L50':L50, 
        'L85':L85, 
    }

# saving data
with open(saving_folder + '/' + 'TestTrainSplitted.json', 'w') as outfile:
    json.dump(result_dict, outfile, indent=2, ensure_ascii=False)