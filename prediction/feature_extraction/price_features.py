import datetime as datetime

import jalali_pandas
import numpy as np
import pandas as pd

#######################
## Calendar Features ##
#######################
def calender_features(ticker):

    df = ticker.copy()

    return_df = pd.DataFrame(
        {
            'jdate': df['date'].jalali.to_jalali()
        }
    )
    return_df = pd.DataFrame(
        {
        'year': return_df['jdate'].jalali.year,
        'quarter': return_df['jdate'].jalali.quarter,
        'month': return_df['jdate'].jalali.month,
        'day': return_df['jdate'].jalali.day,
        'weekday': return_df['jdate'].jalali.weekday,
        }
    )
    
    return return_df

##########################
## Candlestick Features ##
##########################
def candlestick_feature(ticker):

    df = ticker.copy()
    df = pd.concat([df, calender_features(df)], axis = 1)

    def candle_based(df):
        candle_based_df = pd.DataFrame(
            {
                'max_price_1w': df['high'].rolling(5).max(),
                'min_price_1w': df['low'].rolling(5).min(),
                'mean_price_1w': df['adjClose'].rolling(5).mean(),
                'first_price_1w': df['open'].rolling(5).agg(lambda rows: rows.to_numpy()[0]),
                'last_price_1w': df['close'].rolling(5).agg(lambda rows: rows.to_numpy()[-1]),
                'yesterday_price_1w': df['yesterday'].rolling(5).agg(lambda rows: rows.to_numpy()[0]),

                'max_price_2w': df['high'].rolling(10).max(),
                'min_price_2w': df['low'].rolling(10).min(),
                'mean_price_2w': df['adjClose'].rolling(10).mean(),
                'first_price_2w': df['open'].rolling(10).agg(lambda rows: rows.to_numpy()[0]),
                'last_price_2w': df['close'].rolling(10).agg(lambda rows: rows.to_numpy()[-1]),
                'yesterday_price_2w': df['yesterday'].rolling(10).agg(lambda rows: rows.to_numpy()[0]),

                'max_price_3w': df['high'].rolling(15).max(),
                'min_price_3w': df['low'].rolling(15).min(),
                'mean_price_3w': df['adjClose'].rolling(15).mean(),
                'first_price_3w': df['open'].rolling(15).agg(lambda rows: rows.to_numpy()[0]),
                'last_price_3w': df['close'].rolling(15).agg(lambda rows: rows.to_numpy()[-1]),
                'yesterday_price_3w': df['yesterday'].rolling(15).agg(lambda rows: rows.to_numpy()[0]),

                'max_price_1m': df.groupby(['year', 'month'])['high'].transform('max'),
                'min_price_1m': df.groupby(['year', 'month'])['low'].transform('min'),
                'mean_price_1m': df.groupby(['year', 'month'])['adjClose'].transform('mean'),
                'first_price_1m': df.groupby(['year', 'month'])['open'].transform(lambda row: row.to_numpy()[0]),
                'last_price_1m': df.groupby(['year', 'month'])['close'].transform(lambda row: row.to_numpy()[-1]),
                'yesterday_price_1m': df.groupby(['year', 'month'])['yesterday'].transform(lambda row: row.to_numpy()[0]),

                'max_price_3m': df.groupby(['year', 'quarter'])['high'].transform('max'),
                'min_price_3m': df.groupby(['year', 'quarter'])['low'].transform('min'),
                'mean_price_3m': df.groupby(['year', 'quarter'])['adjClose'].transform('mean'),
                'first_price_3m': df.groupby(['year', 'quarter'])['open'].transform(lambda row: row.to_numpy()[0]),
                'last_price_3m': df.groupby(['year', 'quarter'])['close'].transform(lambda row: row.to_numpy()[-1]),
                'yesterday_price_3m': df.groupby(['year', 'quarter'])['yesterday'].transform(lambda row: row.to_numpy()[0]),
            }
        )
        return candle_based_df

    candle_based_df = candle_based(df)

    def shadow_based(df):
        shadow_based_df = pd.DataFrame(
            {
                'shadow_up_1w': (df['max_price_1w'] - np.maximum(df['first_price_1w'], df['last_price_1w'])) / (df['yesterday_price_1w']),
                'shadow_low_1w': (df['min_price_1w'] - np.minimum(df['first_price_1w'], df['last_price_1w'])) / (df['yesterday_price_1w']),
                'body_1w': np.abs(df['last_price_1w'] - df['first_price_1w']) / df['yesterday_price_1w'],
                'mean_to_shadow_up_1w': (df['mean_price_1w'] - df['max_price_1w']) / df['yesterday_price_1w'],
                'mean_to_shadow_low_1w': (df['mean_price_1w'] - df['min_price_1w']) / df['yesterday_price_1w'],

                'shadow_up_2w': (df['max_price_2w'] - np.maximum(df['first_price_2w'], df['last_price_2w'])) / (df['yesterday_price_2w']),
                'shadow_low_2w': (df['min_price_2w'] - np.minimum(df['first_price_2w'], df['last_price_2w'])) / (df['yesterday_price_2w']),
                'body_2w': np.abs(df['last_price_2w'] - df['first_price_2w']) / df['yesterday_price_2w'],
                'mean_to_shadow_up_2w': (df['mean_price_2w'] - df['max_price_2w']) / df['yesterday_price_2w'],
                'mean_to_shadow_low_2w': (df['mean_price_2w'] - df['min_price_2w']) / df['yesterday_price_2w'],

                'shadow_up_3w': (df['max_price_3w'] - np.maximum(df['first_price_3w'], df['last_price_3w'])) / (df['yesterday_price_3w']),
                'shadow_low_3w': (df['min_price_3w'] - np.minimum(df['first_price_3w'], df['last_price_3w'])) / (df['yesterday_price_3w']),
                'body_3w': np.abs(df['last_price_3w'] - df['first_price_3w']) / df['yesterday_price_3w'],
                'mean_to_shadow_up_3w': (df['mean_price_3w'] - df['max_price_3w']) / df['yesterday_price_3w'],
                'mean_to_shadow_low_3w': (df['mean_price_3w'] - df['min_price_3w']) / df['yesterday_price_3w'],

                'shadow_up_1m': (df['max_price_1m'] - np.maximum(df['first_price_1m'], df['last_price_1m'])) / (df['yesterday_price_1m']),
                'shadow_low_1m': (df['min_price_1m'] - np.minimum(df['first_price_1m'], df['last_price_1m'])) / (df['yesterday_price_1m']),
                'body_1m': np.abs(df['last_price_1m'] - df['first_price_1m']) / df['yesterday_price_1m'],
                'mean_to_shadow_up_1m': (df['mean_price_1m'] - df['max_price_1m']) / df['yesterday_price_1m'],
                'mean_to_shadow_low_1m': (df['mean_price_1m'] - df['min_price_1m']) / df['yesterday_price_1m'],

                'shadow_up_3m': (df['max_price_3m'] - np.maximum(df['first_price_3m'], df['last_price_3m'])) / (df['yesterday_price_3m']),
                'shadow_low_3m': (df['min_price_3m'] - np.minimum(df['first_price_3m'], df['last_price_3m'])) / (df['yesterday_price_3m']),
                'body_3m': np.abs(df['last_price_3m'] - df['first_price_3m']) / df['yesterday_price_3m'],
                'mean_to_shadow_up_3m': (df['mean_price_3m'] - df['max_price_3m']) / df['yesterday_price_3m'],
                'mean_to_shadow_low_3m': (df['mean_price_3m'] - df['min_price_3m']) / df['yesterday_price_3m']
            }
        )
        return shadow_based_df

    shadow_based_df = shadow_based(candle_based_df)
    return_df = pd.concat([shadow_based_df, candle_based_df], axis = 1)

    return return_df

###############################################
## Proportion of Prices and Volumes Features ##
###############################################
def prp_based(ticker):

    df = ticker.copy()
    df = pd.concat([df, calender_features(df)], axis = 1)

    return_df = pd.DataFrame(
        {
            'prp_high_1m': df['adjClose'] / df.groupby(['year', 'month'])['high'].transform('max'),
            'prp_high_1m': df['adjClose'] / df.groupby(['year', 'month'])['low'].transform('min'),

            'prp_high_3m': df['adjClose'] / df.groupby(['year', 'quarter'])['high'].transform('max'),
            'prp_high_3m': df['adjClose'] / df.groupby(['year', 'quarter'])['low'].transform('min'),

            'prp_value_1w1m': df['value'].rolling(5).mean() / df.groupby(['year', 'quarter'])['value'].transform('mean'),
            'prp_value_2w3m': df['value'].rolling(10).mean() / df.groupby(['year', 'quarter'])['value'].transform('mean'),
            'prp_value_3w3m': df['value'].rolling(15).mean() / df.groupby(['year', 'quarter'])['value'].transform('mean'),
            'prp_value_1m3m': df.groupby(['year', 'month'])['value'].transform('mean') / df.groupby(['year', 'quarter'])['value'].transform('mean')
        }
    )
    
    return return_df

######################
## Returns Features ##
######################
def ret_based(ticker):

    df = ticker.copy()

    return_df = pd.DataFrame(
        {
            'ret_1d':((df['adjClose']- df['yesterday']) / df['yesterday']),
            'ret_3d':((df['adjClose']- df['yesterday']) / df['yesterday']).rolling(3).sum(),
            'ret_1w':((df['adjClose']- df['yesterday']) / df['yesterday']).rolling(7).sum(),
            'ret_2w':((df['adjClose']- df['yesterday']) / df['yesterday']).rolling(14).sum(),
            'ret_1m':((df['adjClose']- df['yesterday']) / df['yesterday']).rolling(30).sum(),
            'ret_3m':((df['adjClose']- df['yesterday']) / df['yesterday']).rolling(90).sum(),

            'ret_1d_log':(np.log(df['adjClose']) - np.log(df['yesterday'])),
            'ret_3d_log':(np.log(df['adjClose']) - np.log(df['yesterday'])).rolling(3).sum(),
            'ret_1w_log':(np.log(df['adjClose']) - np.log(df['yesterday'])).rolling(7).sum(),
            'ret_2w_log':(np.log(df['adjClose']) - np.log(df['yesterday'])).rolling(14).sum(),
            'ret_1m_log':(np.log(df['adjClose']) - np.log(df['yesterday'])).rolling(30).sum(),
            'ret_3m_log':(np.log(df['adjClose']) - np.log(df['yesterday'])).rolling(90).sum(),

            'buy_queue_locked' : np.logical_and(np.isclose(df['high'], df['low']), df['high'] > df['yesterday']) * 1,
            'sell_queue_locked' : np.logical_and(np.isclose(df['high'], df['low']), df['low'] < df['yesterday']) * 1
        }
    )
    return return_df

#######################
## Shift Prices Data ##
#######################
def shift_data(ticker):

    df = ticker.copy()

    return_df = pd.concat(
        [
            df.shift(1).drop(columns = ['date', 'jdate']).add_prefix('last1d_'),
            df.shift(5).drop(columns = ['date', 'jdate']).add_prefix('last5d_')
        ], axis = 1  
    )

    return return_df

#############################
## Weighted Value Features ##
#############################
def weight_feature(ticker):

    df = ticker.copy()

    def value_based(df):
        value_based_df = pd.DataFrame(
            {
                'value_20d': df['value'].rolling(20).mean(),
                'value_30d': df['value'].rolling(30).mean(),
                'value_40d': df['value'].rolling(40).mean(),
                'value_50d': df['value'].rolling(50).mean(),
                'value_60d': df['value'].rolling(60).mean(),
                'value_70d': df['value'].rolling(70).mean(),
                'value_80d': df['value'].rolling(80).mean(),
                'value_90d': df['value'].rolling(90).mean(),
                'value_100d': df['value'].rolling(100).mean(),
            }
        )
        return value_based_df
        
    value_based_df = value_based(df)
    temp_value_based_df = pd.concat([df['value'], value_based_df], axis = 1)
    
    def weight_based(df):
        weight_based_df = pd.DataFrame(
            {
                'value_weight': df['value'] / df['value'].sum(),
                'value_weight_20d': df['value_20d'] / df['value_20d'].sum(),
                'value_weight_30d': df['value_30d'] / df['value_30d'].sum(),
                'value_weight_40d': df['value_40d'] / df['value_40d'].sum(),
                'value_weight_50d': df['value_50d'] / df['value_50d'].sum(),
                'value_weight_60d': df['value_60d'] / df['value_60d'].sum(),
                'value_weight_70d': df['value_70d'] / df['value_70d'].sum(),
                'value_weight_80d': df['value_80d'] / df['value_80d'].sum(),
                'value_weight_90d': df['value_90d'] / df['value_90d'].sum(),
                'value_weight_100d': df['value_100d'] / df['value_100d'].sum(),
            }
        )
        return weight_based_df

    weight_based_df = weight_based(temp_value_based_df)
    return_df = pd.concat([value_based_df, weight_based_df], axis = 1)

    return return_df