import datetime as datetime
import jalali_pandas
import numpy as np
import pandas as pd


## calender features ##
#######################
def calender_features(stock):
    df = stock
    
    return_df = pd.DataFrame(
        {
            'jdate': df['date'].jalali.to_jalali()
        }
    )
    return_df = pd.DataFrame(
        {
        'year': return_df['jdate'].jalali.year,
        'month': return_df['jdate'].jalali.month,
        'quarter': return_df['jdate'].jalali.quarter,
        'day': return_df['jdate'].jalali.day,
        'weekday': return_df['jdate'].jalali.weekday,
        }
    )
    
    return return_df


## candlestick features ##
##########################
def candle_based(stock):
    df = stock

    return_df = pd.DataFrame(
        {
        'max_price7d': df['high'].rolling(5).max(),
        'min_price7d': df['low'].rolling(5).min(),
        'first_price7d': df['open'].rolling(5).agg(lambda rows: rows[0]),
        'last_price7d': df['close'].rolling(5).agg(lambda rows: rows[-1]),
        'yesterday_price7d': df['yesterday'].rolling(5).agg(lambda rows: rows[0]),
        'max_price14d': df['high'].rolling(10).max(),
        'min_price14d': df['low'].rolling(10).min(),
        'first_price14d': df['open'].rolling(10).agg(lambda rows: rows[0]),
        'last_price14d': df['close'].rolling(10).agg(lambda rows: rows[-1]),
        'yesterday_price14d': df['yesterday'].rolling(10).agg(lambda rows: rows[0]),
        'max_price21d': df['high'].rolling(15).max(),
        'min_price21d': df['low'].rolling(15).min(),
        'first_price21d': df['open'].rolling(15).agg(lambda rows: rows[0]),
        'last_price21d': df['close'].rolling(15).agg(lambda rows: rows[-1]),
        'yesterday_price21d': df['yesterday'].rolling(15).agg(lambda rows: rows[0]),
        'max_price30d': df['high'].rolling(23).max(),
        'min_price30d': df['low'].rolling(23).min(),
        'first_price30d': df['open'].rolling(23).agg(lambda rows: rows[0]),
        'last_price30d': df['close'].rolling(23).agg(lambda rows: rows[-1]),
        'yesterday_price30d': df['yesterday'].rolling(23).agg(lambda rows: rows[0]),
        'max_price60d': df['high'].rolling(44).max(),
        'min_price60d': df['low'].rolling(44).min(),
        'first_price60d': df['open'].rolling(44).agg(lambda rows: rows[0]),
        'last_price60d': df['close'].rolling(44).agg(lambda rows: rows[-1]),
        'yesterday_price60d': df['yesterday'].rolling(44).agg(lambda rows: rows[0]),
        'max_price90d': df['high'].rolling(70).max(),
        'min_price90d': df['low'].rolling(70).min(),
        'first_price90d': df['open'].rolling(70).agg(lambda rows: rows[0]),
        'last_price90d': df['close'].rolling(70).agg(lambda rows: rows[-1]),
        'yesterday_price90d': df['yesterday'].rolling(70).agg(lambda rows: rows[0])
        }
    )

    return return_df
    
    
def ret_based_func(stock):
    df = stock
    return_df = pd.DataFrame({\
                              'ret1d_log':(np.log(df.adjClose) - np.log(df.yesterday)), \
                              'ret3d_log':(np.log(df.adjClose) - np.log(df.yesterday)).rolling(3).sum(), \
                              'ret7d_log':(np.log(df.adjClose) - np.log(df.yesterday)).rolling(7).sum(), \
                              'ret14d_log':(np.log(df.adjClose) - np.log(df.yesterday)).rolling(14).sum(), \
                              'ret30d_log':(np.log(df.adjClose) - np.log(df.yesterday)).rolling(30).sum(), \
                              'ret60d_log':(np.log(df.adjClose) - np.log(df.yesterday)).rolling(60).sum(), \
                              'ret90d_log':(np.log(df.adjClose) - np.log(df.yesterday)).rolling(90).sum(), \
                              'ret120d_log':(np.log(df.adjClose) - np.log(df.yesterday)).rolling(120).sum(), \
                              'ret300d_log':(np.log(df.adjClose) - np.log(df.yesterday)).rolling(300).sum(), \
                              'lastclose_log':(np.log(df['adjClose']) - np.log(df['close'])), \
                              'buy_queue_locked' : np.logical_and(np.isclose(df['high'], df['low']), df['high'] > df['yesterday']), \
                              'sell_queue_locked' : np.logical_and(np.isclose(df['high'], df['low']), df['low'] < df['yesterday']), \
                              })
    return return_df

def prp_based_func(stock):
    df = stock
    df.index = [pd.Timestamp(item) for item in df['date']]
    return_df = pd.DataFrame({\
        'prp_high30d':df['adjClose']/df['high'].rolling('30d').max(), \
        'prp_high60d':df['adjClose']/df['high'].rolling('60d').max(), \
        'prp_high90d':df['adjClose']/df['high'].rolling('90d').max(), \
        'prp_high120d':df['adjClose']/df['high'].rolling('120d').max(), \
        'prp_high300d':df['adjClose']/df['high'].rolling('300d').max(), \
        'prp_low30d':df['adjClose']/df['low'].rolling('30d').min(), \
        'prp_low60d':df['adjClose']/df['low'].rolling('60d').min(), \
        'prp_low90d':df['adjClose']/df['low'].rolling('90d').min(), \
        'prp_low120d':df['adjClose']/df['low'].rolling('120d').min(), \
        'prp_low300d':df['adjClose']/df['low'].rolling('300d').min(), \
        'prp_value3d30d':df['value'].rolling('3d').mean()/df['value'].rolling('30d').mean(), \
        'prp_value5d60d':df['value'].rolling('5d').mean()/df['value'].rolling('60d').mean(), \
        'prp_value15d120d':df['value'].rolling('15d').mean()/df['value'].rolling('120d').mean(), \
        'prp_value30d200d':df['value'].rolling('30d').mean()/df['value'].rolling('200d').mean(), \
        })
    return return_df

def wght_based_feature(foladHist):
    print('weight based features')
    
    def value20d_based_func(foladHist):
        return_df = pd.DataFrame({\
            'value_20d':foladHist['value'].rolling(20).mean(), \
            })
        return return_df
    
    df_value20_data = pd.concat([foladHist[['date','value']], value20d_based_func(foladHist)], axis=1)
    
    def wght_based_func(stock_date_group):
        return_df = pd.DataFrame({\
            'value_weight':foladHist['value']/foladHist['value'].sum(), \
            'value_weight20d':foladHist['value_20d']/foladHist['value_20d'].sum(), \
            })
        return return_df

    df_wght_based = value20d_based_func(foladHist)


    df_feature = pd.concat([df_value20_data, df_wght_based], axis=1)
    return df_feature


