#Daily Features

#Import
import pandas as pd
import numpy as np
import datetime as datetime



# ## candlestick feature ##
# #########################
def candlestick_feature(df):
    def candle_based(df):
        candle_based_df = pd.DataFrame({\
            'max_price7d': df['high'].rolling(5).max(),\
            'min_price7d': df['low'].rolling(5).min(),\
            'ave_price7d': df['low'].rolling(5).mean(),\
            'first_price7d': df['open'].rolling(5).agg(lambda rows: rows.to_numpy()[0]),\
            'last_price7d': df['close'].rolling(5).agg(lambda rows: rows.to_numpy()[-1]),\
            'yesterday_price7d': df['yesterday'].rolling(5).agg(lambda rows: rows.to_numpy()[0]),\
            'max_price14d': df['high'].rolling(10).max(),\
            'min_price14d': df['low'].rolling(10).min(),\
            'ave_price7d': df['low'].rolling(10).mean(),\
            'first_price14d': df['open'].rolling(10).agg(lambda rows: rows.to_numpy()[0]),\
            'last_price14d': df['close'].rolling(10).agg(lambda rows: rows.to_numpy()[-1]),\
            'yesterday_price14d': df['yesterday'].rolling(10).agg(lambda rows: rows.to_numpy()[0]),\
            'max_price21d': df['high'].rolling(15).max(),\
            'min_price21d': df['low'].rolling(15).min(),\
            'ave_price7d': df['low'].rolling(15).mean(),\
            'first_price21d': df['open'].rolling(15).agg(lambda rows: rows.to_numpy()[0]),\
            'last_price21d': df['close'].rolling(15).agg(lambda rows: rows.to_numpy()[-1]),\
            'yesterday_price21d': df['yesterday'].rolling(15).agg(lambda rows: rows.to_numpy()[0]),\
            'max_price30d': df['high'].rolling(23).max(),\
            'min_price30d': df['low'].rolling(23).min(),\
            'ave_price7d': df['low'].rolling(23).mean(),\
            'first_price30d': df['open'].rolling(23).agg(lambda rows: rows.to_numpy()[0]),\
            'last_price30d': df['close'].rolling(23).agg(lambda rows: rows.to_numpy()[-1]),\
            'yesterday_price30d': df['yesterday'].rolling(23).agg(lambda rows: rows.to_numpy()[0]),\
            'max_price60d': df['high'].rolling(44).max(),\
            'min_price60d': df['low'].rolling(44).min(),\
            'ave_price7d': df['low'].rolling(44).mean(),\
            'first_price60d': df['open'].rolling(44).agg(lambda rows: rows.to_numpy()[0]),\
            'last_price60d': df['close'].rolling(44).agg(lambda rows: rows.to_numpy()[-1]),\
            'yesterday_price60d': df['yesterday'].rolling(44).agg(lambda rows: rows.to_numpy()[0]),\
            'max_price90d': df['high'].rolling(70).max(),\
            'min_price90d': df['low'].rolling(70).min(),\
            'ave_price7d': df['low'].rolling(70).mean(),\
            'first_price90d': df['open'].rolling(70).agg(lambda rows: rows.to_numpy()[0]),\
            'last_price90d': df['close'].rolling(70).agg(lambda rows: rows.to_numpy()[-1]),\
            'yesterday_price90d': df['yesterday'].rolling(70).agg(lambda rows: rows.to_numpy()[0]),\
            })
        return candle_based_df
    df_candle_based = candle_based(df)


    def shadow_based(df):
        shadow_based_df = pd.DataFrame({\
            'shadow_up7d': (df['max_price7d'] - np.maximum(df['first_price7d'],df['last_price7d']))/(df['yesterday_price7d']),\
            'shadow_low7d': (df['min_price7d'] - np.minimum(df['first_price7d'],df['last_price7d']))/(df['yesterday_price7d']),\
            'body7d': np.abs(df['last_price7d']-df['first_price7d'])/df['yesterday_price7d'],\
            'shadow_up14d': (df['max_price14d'] - np.maximum(df['first_price14d'],df['last_price14d']))/(df['yesterday_price14d']),\
            'shadow_low14d': (df['min_price14d'] - np.minimum(df['first_price14d'],df['last_price14d']))/(df['yesterday_price14d']),\
            'body14d': np.abs(df['last_price14d']-df['first_price14d'])/df['yesterday_price14d'],\
            'shadow_up21d': (df['max_price21d'] - np.maximum(df['first_price21d'],df['last_price21d']))/(df['yesterday_price21d']),\
            'shadow_low21d': (df['min_price21d'] - np.minimum(df['first_price21d'],df['last_price21d']))/(df['yesterday_price21d']),\
            'body21d': np.abs(df['last_price21d']-df['first_price21d'])/df['yesterday_price21d'],\
            'shadow_up30d': (df['max_price30d'] - np.maximum(df['first_price30d'],df['last_price30d']))/(df['yesterday_price30d']),\
            'shadow_low30d': (df['min_price30d'] - np.minimum(df['first_price30d'],df['last_price30d']))/(df['yesterday_price30d']),\
            'body30d': np.abs(df['last_price30d']-df['first_price30d'])/df['yesterday_price30d'],\
            'shadow_up60d': (df['max_price60d'] - np.maximum(df['first_price60d'],df['last_price60d']))/(df['yesterday_price60d']),\
            'shadow_low60d': (df['min_price60d'] - np.minimum(df['first_price60d'],df['last_price60d']))/(df['yesterday_price60d']),\
            'body60d': np.abs(df['last_price60d']-df['first_price60d'])/df['yesterday_price60d'],\
            'shadow_up90d': (df['max_price90d'] - np.maximum(df['first_price90d'],df['last_price90d']))/(df['yesterday_price90d']),\
            'shadow_low90d': (df['min_price90d'] - np.minimum(df['first_price90d'],df['last_price90d']))/(df['yesterday_price90d']),\
            'body90d': np.abs(df['last_price90d']-df['first_price90d'])/df['yesterday_price90d']\
            })
        return shadow_based_df
    df_shadow_based = shadow_based(df_candle_based)

    df_feature = pd.concat([df_shadow_based, df_candle_based], axis=1)
    return df_feature



# ## weighted feature ##
# ######################
def weight_feature(df):
    def value_based(df):
        value_based_df = pd.DataFrame({\
            'value_20d':df['value'].rolling(20).mean(), \
            'value_30d':df['value'].rolling(30).mean(), \
            'value_40d':df['value'].rolling(40).mean(), \
            'value_50d':df['value'].rolling(50).mean(), \
            'value_60d':df['value'].rolling(60).mean(), \
            'value_70d':df['value'].rolling(70).mean(), \
            'value_80d':df['value'].rolling(80).mean(), \
            'value_90d':df['value'].rolling(90).mean(), \
            'value_100d':df['value'].rolling(100).mean(), \
            })
        return value_based_df
    df_value_based = pd.concat([df['value'], value_based(df)], axis=1)
    
    def weight_based(df):
        weight_based_df = pd.DataFrame({\
            'value_weight':df['value']/df['value'].sum(), \
            'value_weight20d':df['value_20d']/df['value_20d'].sum(), \
            'value_weight30d':df['value_30d']/df['value_30d'].sum(), \
            'value_weight40d':df['value_40d']/df['value_40d'].sum(), \
            'value_weight50d':df['value_50d']/df['value_50d'].sum(), \
            'value_weight60d':df['value_60d']/df['value_60d'].sum(), \
            'value_weight70d':df['value_70d']/df['value_70d'].sum(), \
            'value_weight80d':df['value_80d']/df['value_80d'].sum(), \
            'value_weight90d':df['value_90d']/df['value_90d'].sum(), \
            'value_weight100d':df['value_100d']/df['value_100d'].sum(), \
            })
        return weight_based_df
    df_weight_based = value_based(df_value_based)


    df_feature = pd.concat([df_value_based, df_weight_based], axis=1)
    return df_feature




## proportion price in different time frames ##
###############################################
def proportion_feature(df):
    df_feature = pd.DataFrame(
        {
            'prp_high30d': df['adjClose']/df['high'].rolling(30).max(),
            'prp_high60d': df['adjClose']/df['high'].rolling(60).max(),
            'prp_high90d': df['adjClose']/df['high'].rolling(90).max(),
            'prp_low30d': df['adjClose']/df['low'].rolling(30).min(),
            'prp_low60d': df['adjClose']/df['low'].rolling(60).min(),
            'prp_low90d': df['adjClose']/df['low'].rolling(90).min(),
            'prp_value3d30d': df['value'].rolling(3).mean()/df['value'].rolling(30).mean(),
            'prp_value5d60d': df['value'].rolling(5).mean()/df['value'].rolling(60).mean(),
        }
    )
    return df_feature




## logarithmic returns features ##
##################################
def logarithmic_feature(df):
    df_feature = pd.DataFrame(
        {
            'ret1d_log':(np.log(df['adjClose']) - np.log(df['yesterday'])),
            'ret3d_log':(np.log(df['adjClose']) - np.log(df['yesterday'])).rolling(3).sum(),
            'ret7d_log':(np.log(df['adjClose']) - np.log(df['yesterday'])).rolling(7).sum(),
            'ret14d_log':(np.log(df['adjClose']) - np.log(df['yesterday'])).rolling(14).sum(),
            'ret30d_log':(np.log(df['adjClose']) - np.log(df['yesterday'])).rolling(30).sum(),
            'ret60d_log':(np.log(df['adjClose']) - np.log(df['yesterday'])).rolling(60).sum(),
            'lastclose_log':(np.log(df['adjClose']) - np.log(df['close'])),
            'buy_queue_locked' : np.logical_and(np.isclose(df['high'], df['low']), df['high'] > df['yesterday']),
            'sell_queue_locked' : np.logical_and(np.isclose(df['high'], df['low']), df['low'] < df['yesterday'])
        }
    )
    return df_feature




## shift data for 1 day ##
##########################
def shift_data(df):
    df_feature = pd.concat(
        [
            df.shift(1).drop(columns = 'date').add_prefix('last1d_')
        ], axis = 1  
    )
    return df_feature



