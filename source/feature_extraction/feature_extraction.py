from .daily_feature_extraction import candle_based_feature, wght_based_feature, price_based_feature, proportion_based_feature, indv_nonindv_based_feature, \
    calender_based_feature, bwd_prc_based_feature, fwd_prc_based_feature, main_feature
from .daily_technical_indicator import sma_based_features, ichimoku_based_feature, cci_based_feature, rvi_based_feature, roc_based_feature, \
    mfi_based_feature, cci_light_based_feature, ema_light_based_feature, ema_based_feature, aroon_based_feature, \
    rsi_based_feature, uo_based_feature, stoch_based_feature, obv_based_feature, macd_based_feature
from .group_feature_extraction import group_based_features, dollar_group_based_features
from .overall_weighted_aggregation import level_weight_features_edited

import pandas as pd
import os as os
import json as json

# constants
FEATURE_DATA_SUFFIX = '.csv'
FEATURE_DETAIL_SUFFIX = '.json'
FEATURE_REPORT_SUFFIX = '.json'
FEATURE_REPORT_FILENAME = 'daily_feature_report'
FEATURE_DATA_FILENAME = 'daily_feature_data'
FEATURE_DETAIL_FILENAME = 'daily_feature_detail'

OVERALL_FEATURE_DATA_SUFFIX = '.csv'
OVERALL_FEATURE_DETAIL_SUFFIX = '.json'
OVERALL_FEATURE_REPORT_SUFFIX = '.json'
OVERALL_FEATURE_REPORT_FILENAME = 'overall_feature_report'
OVERALL_FEATURE_DATA_FILENAME = 'overall_feature_data'
OVERALL_FEATURE_DETAIL_FILENAME = 'overall_feature_detail'

def stock_daily_feature_extraction_procedure(stock_daily_data, market_index_data, stock_daily_data_with_groups):
    feature_data = pd.DataFrame([])
    feature_detail = dict()

    # price based features
    main_data, main_detail = main_feature(stock_daily_data) 
    # clnd_data, clnd_detail = calender_based_feature(stock_daily_data)
    wght_data, wght_detail = wght_based_feature(stock_daily_data)
    prc_data, prc_detail = price_based_feature(stock_daily_data)
    prp_data, prp_detail = proportion_based_feature(stock_daily_data)
    indv_nonindv_data, indv_nonindv_detail = indv_nonindv_based_feature(stock_daily_data, wght_data, prp_data)
    candle_data, candle_detail = candle_based_feature(stock_daily_data)

    # technical features
    sma_data, sma_detail = sma_based_features(stock_daily_data)
    ich_data, ich_detail = ichimoku_based_feature(stock_daily_data)
    cci_data, cci_detail = cci_based_feature(stock_daily_data)
    rvi_data, rvi_detail = rvi_based_feature(stock_daily_data)
    roc_data, roc_detail = roc_based_feature(stock_daily_data)
    mfi_data, mfi_detail = mfi_based_feature(stock_daily_data)
    ema_data, ema_detail = ema_based_feature(stock_daily_data)
    aroon_data, aroon_detail = aroon_based_feature(stock_daily_data)
    rsi_data, rsi_detail = rsi_based_feature(stock_daily_data)
    uo_data, uo_detail = uo_based_feature(stock_daily_data)
    stoch_data, stoch_detail = stoch_based_feature(stock_daily_data)
    obv_data, obv_detail = obv_based_feature(stock_daily_data)
    macd_data, macd_detail = macd_based_feature(stock_daily_data)
    cci_light_data, cci_light_detail = cci_light_based_feature(stock_daily_data)
    ema_light_data, ema_light_detail = ema_light_based_feature(stock_daily_data)

    # merging dataframes
    merged_data_df = stock_daily_data.merge(market_index_data, how='left', on='date')
    
    # backward features
    bwd_prc_data, bwd_prc_detail = bwd_prc_based_feature(merged_data_df)
    
    # forward features
    fwd_prc_data, fwd_prc_detail = fwd_prc_based_feature(merged_data_df)
    
    # group features
    daily_group_data, daily_group_detail = group_based_features(stock_daily_data_with_groups)
    daily_dollar_data, daily_dollar_detail = dollar_group_based_features(stock_daily_data_with_groups)

    # feature merging
    feature_data = pd.concat([\
        main_data, wght_data, prc_data, prp_data, indv_nonindv_data, candle_data,\
        sma_data, ich_data, cci_data, rvi_data, roc_data, mfi_data, cci_light_data, ema_data, aroon_data, rsi_data, \
        uo_data, stoch_data, obv_data, macd_data, ema_light_data, bwd_prc_data, fwd_prc_data, daily_group_data,\
            daily_dollar_data], axis=1)
    for d in [\
        main_detail, wght_detail, prc_detail, prp_detail, indv_nonindv_detail, candle_detail, \
        sma_detail, ich_detail, cci_detail, rvi_detail, roc_detail, mfi_detail, cci_light_detail, ema_detail, aroon_detail, \
        rsi_detail, uo_detail, stoch_detail, obv_detail, macd_detail, ema_light_detail, bwd_prc_detail, fwd_prc_detail, daily_group_detail,\
            daily_dollar_detail]:
        feature_detail.update(d)
    return feature_data, feature_detail

def overall_feature_extraction_procedure(stock_daily_feature, market_index_data):
    feature_data = pd.DataFrame([])
    feature_detail = dict()

    # price based features
    level_weight_data = level_weight_features_edited(stock_daily_feature) 

    # feature merging
    feature_data = level_weight_data
    # for d in [\
    #     level_weight_detail]:
    #     feature_detail.update(d)
    return feature_data

def stock_daily_feature_extraction(saving_directory, clean_daily_data_directory, clean_index_data_directory, \
    group_info_directory, group_info_filename = 'stock_ids',group_info_suffix = '.csv' ,\
    clean_daily_data_filename='cleaned_daily_all', clean_daily_data_suffix='.csv', \
    clean_index_data_filename='cleaned_index', clean_index_data_suffix='.csv'):
    feature_data = pd.DataFrame([])
    feature_detail = dict()
    report_dict = dict()

    # exit if there does not exist any saving directory
    if (saving_directory is None) or (os.path.isdir(saving_directory) is False):
        print('Saving directory does not exist.')
        report_dict['directory_error'] = 'Saving directory does not exist.'
        return report_dict

    # csv read
    csv_daily_file_path = clean_daily_data_directory + '/' + clean_daily_data_filename + clean_daily_data_suffix
    stock_daily_data = pd.read_csv(csv_daily_file_path).drop(['Unnamed: 0'], axis=1)

    csv_index_file_path = clean_index_data_directory + '/' + clean_index_data_filename + clean_index_data_suffix
    market_index_data = pd.read_csv(csv_index_file_path).drop(['Unnamed: 0'], axis=1)

    csv_group_info_path = group_info_directory + '/' + group_info_filename + group_info_suffix
    stock_info = pd.read_csv(csv_group_info_path)
    stock_info = stock_info.rename(columns={"symbol":"stock_name"})
    stock_daily_data_with_groups = stock_daily_data.merge(stock_info, on='stock_name', how='left')

    # feature extraction procedure
    feature_data, feature_detail = stock_daily_feature_extraction_procedure(stock_daily_data, market_index_data, stock_daily_data_with_groups)

    # save the results
    data_saving_path = saving_directory + '/' + FEATURE_DATA_FILENAME + FEATURE_DATA_SUFFIX
    try:
        feature_data.to_csv(data_saving_path, sep=',', line_terminator=None)
    except:
        error_msg = 'Some error in saving features.'
        print(error_msg)
        report_dict['file_error'] = error_msg
    
    # saving detail file
    saving_path = saving_directory + '/' + FEATURE_DETAIL_FILENAME + FEATURE_DETAIL_SUFFIX
    with open(saving_path, 'w') as outfile:
        json.dump(feature_detail, outfile, indent=2, ensure_ascii=False)

    # saving report file
    saving_path = saving_directory + '/' + FEATURE_REPORT_FILENAME + FEATURE_REPORT_SUFFIX
    with open(saving_path, 'w') as outfile:
        json.dump(report_dict, outfile, indent=2, ensure_ascii=False)
    
    return report_dict

def overal_feature_extraction(saving_directory, feature_daily_data_directory, clean_index_data_directory, \
    group_info_directory, group_info_filename = 'stock_ids',group_info_suffix = '.csv' ,\
    feature_daily_data_filename='daily_feature_data', feature_daily_data_suffix='.csv', \
    clean_index_data_filename='cleaned_index', clean_index_data_suffix='.csv'):
    feature_data = pd.DataFrame([])
    # feature_detail = dict()
    report_dict = dict()

    # exit if there does not exist any saving directory
    if (saving_directory is None) or (os.path.isdir(saving_directory) is False):
        print('Saving directory does not exist.')
        report_dict['directory_error'] = 'Saving directory does not exist.'
        return report_dict

    # csv read
    csv_daily_file_path = feature_daily_data_directory + '/' + feature_daily_data_filename + feature_daily_data_suffix
    stock_daily_feature = pd.read_csv(csv_daily_file_path).drop(['Unnamed: 0'], axis=1)

    csv_index_file_path = clean_index_data_directory + '/' + clean_index_data_filename + clean_index_data_suffix
    market_index_data = pd.read_csv(csv_index_file_path).drop(['Unnamed: 0'], axis=1)

    csv_group_info_path = group_info_directory + '/' + group_info_filename + group_info_suffix
    stock_info = pd.read_csv(csv_group_info_path)
    stock_info = stock_info.rename(columns={"symbol":"stock_name"})
    stock_info = stock_info[['stock_name', 'level']]
    stock_daily_feature = stock_daily_feature.merge(stock_info, on='stock_name', how='left')

    # feature extraction procedure
    feature_data = overall_feature_extraction_procedure(stock_daily_feature, market_index_data)

    # save the results
    data_saving_path = saving_directory + '/' + OVERALL_FEATURE_DATA_FILENAME + OVERALL_FEATURE_DATA_SUFFIX
    try:
        feature_data.to_csv(data_saving_path, sep=',', line_terminator=None)
    except:
        error_msg = 'Some error in saving features.'
        print(error_msg)
        report_dict['file_error'] = error_msg
    
    # saving detail file
    # saving_path = saving_directory + '/' + OVERALL_FEATURE_DETAIL_FILENAME + OVERALL_FEATURE_DETAIL_SUFFIX
    # with open(saving_path, 'w') as outfile:
    #     json.dump(feature_detail, outfile, indent=2, ensure_ascii=False)

    # saving report file
    saving_path = saving_directory + '/' + OVERALL_FEATURE_REPORT_FILENAME + OVERALL_FEATURE_REPORT_SUFFIX
    with open(saving_path, 'w') as outfile:
        json.dump(report_dict, outfile, indent=2, ensure_ascii=False)
    
    return report_dict