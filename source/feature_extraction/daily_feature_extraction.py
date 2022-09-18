import pandas as pd
import numpy as np
import datetime as datetime
import jdatetime as jdatetime
import swifter as swifter

# TODO: time-based convolution

def main_feature(stock_daily_data):
    print('main features')
    feature_data = pd.DataFrame([])
    feature_detail = dict()

    # features
    feature_data = stock_daily_data[['date', 'stock_name', 'max_price', 'min_price', 'close_price', 'last_price',
       'first_price', 'yesterday_price', 'value', 'volume', 'count',
       'Individual_buy_count', 'NonIndividual_buy_count',
       'Individual_sell_count', 'NonIndividual_sell_count',
       'Individual_buy_volume', 'NonIndividual_buy_volume',
       'Individual_sell_volume', 'NonIndividual_sell_volume',
       'Individual_buy_value', 'NonIndividual_buy_value',
       'Individual_sell_value', 'NonIndividual_sell_value', 'adj_max_price',
       'adj_min_price', 'adj_first_price', 'adj_last_price', 'adj_volume',
       'adj_close_price']]
    feature_detail = {\
        'date':{'description':'date'}, \
        'stock_name':{'description':'stock_name'}, \
        'max_price':{'description':'max_price'}, \
        'min_price':{'description':'min_price'}, \
        'close_price':{'description':'close_price'}, \
        'last_price':{'description':'last_price'}, \
        'first_price':{'description':'first_price'}, \
        'yesterday_price':{'description':'yesterday_price'}, \
        'value':{'description':'value'}, \
        'volume':{'description':'volume'}, \
        'count':{'description':'count'}, \
        'Individual_buy_count':{'description':'Individual_buy_count'}, \
        'NonIndividual_buy_count':{'description':'NonIndividual_buy_count'}, \
        'Individual_sell_count':{'description':'Individual_sell_count'}, \
        'NonIndividual_sell_count':{'description':'NonIndividual_sell_count'}, \
        'Individual_buy_volume':{'description':'Individual_buy_volume'}, \
        'NonIndividual_buy_volume':{'description':'NonIndividual_buy_volume'}, \
        'Individual_sell_volume':{'description':'Individual_sell_volume'}, \
        'NonIndividual_sell_volume':{'description':'NonIndividual_sell_volume'}, \
        'Individual_buy_value':{'description':'Individual_buy_value'}, \
        'NonIndividual_buy_value':{'description':'NonIndividual_buy_value'}, \
        'Individual_sell_value':{'description':'Individual_sell_value'}, \
        'NonIndividual_sell_value':{'description':'NonIndividual_sell_value'}, \
        'adj_max_price':{'description':'adj_max_price'}, \
        'adj_min_price':{'description':'adj_min_price'}, \
        'adj_first_price':{'description':'adj_first_price'}, \
        'adj_last_price':{'description':'adj_last_price'}, \
        'adj_volume':{'description':'adj_volume'}, \
        'adj_close_price':{'description':'adj_close_price'}, \
        }

    return feature_data, feature_detail

def wght_based_feature(stock_daily_data):
    print('weight based features')
    feature_data = pd.DataFrame([])
    feature_detail = dict()

    # grouping stock
    stock_groups = stock_daily_data.groupby('stock_name')

    # map function
    def value20d_based_func(stock_group):
        initial_index = stock_group.index
        df = stock_group
        df.index = [pd.Timestamp(item) for item in df['date']]
        return_df = pd.DataFrame({\
            'value_20d':df['value'].rolling('20d').mean(), \
            })
        return_df.index = initial_index
        return return_df
    
    value20d_data = stock_groups.apply(value20d_based_func).reset_index(drop=True)
    value20d_detail = {\
            'value_20d':{'description':'mean value of stock in recent 20 days.'}, \
            }

    value_data = pd.concat([stock_daily_data[['date','stock_name','value']], value20d_data], axis=1)
    
    # grouping date
    stock_groups = value_data.groupby('date')

    # map function
    def wght_based_func(stock_date_group):
        initial_index = stock_date_group.index
        df = stock_date_group
        df.index = [pd.Timestamp(item) for item in df['date']]
        return_df = pd.DataFrame({\
            'value_weight':df['value']/df['value'].sum(), \
            'value_weight20d':df['value_20d']/df['value_20d'].sum(), \
            })
        return_df.index = initial_index
        return return_df
    
    feature_data = stock_groups.apply(wght_based_func).reset_index(drop=True)
    feature_detail = {\
            'value_weight':{'description':'ratio of value to total value of market at each day.'}, \
            'value_weight20d':{'description':'ratio of value in recent 20 days to total value of market at each day.'}, \
            }

    return feature_data, feature_detail

def price_based_feature(stock_daily_data):
    print('price based features')
    feature_data = pd.DataFrame([])
    feature_detail = dict()

    # grouping
    stock_groups = stock_daily_data.groupby('stock_name')

    # map function
    # Yesterday price adjusts when capital increase happens. Thus, there won't be a problem with capital increase.
    def ret_based_func(stock_group):
        initial_index = stock_group.index
        df = stock_group
        df.index = [pd.Timestamp(item) for item in df['date']]
        return_df = pd.DataFrame({\
            'ret1d_log':(np.log(df['close_price']) - np.log(df['yesterday_price'])), \
            'ret3d_log':(np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('3d').sum(), \
            'ret7d_log':(np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('7d').sum(), \
            'ret14d_log':(np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('14d').sum(), \
            'ret30d_log':(np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('30d').sum(), \
            'ret60d_log':(np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('60d').sum(), \
            'ret90d_log':(np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('90d').sum(), \
            'ret120d_log':(np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('120d').sum(), \
            'ret300d_log':(np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('300d').sum(), \
            'lastclose_log':(np.log(df['last_price']) - np.log(df['close_price'])), \
            'buy_queue_locked' : np.logical_and(np.isclose(df['max_price'], df['min_price']), df['max_price'] > df['yesterday_price']), \
            'sell_queue_locked' : np.logical_and(np.isclose(df['max_price'], df['min_price']), df['min_price'] < df['yesterday_price']), \
            })
        return_df.index = initial_index
        return return_df
    
    feature_data = stock_groups.apply(ret_based_func).reset_index(drop=True)
    feature_detail = {\
            'ret1d_log':{'description':'return per day with respect to 1 day before.'}, \
            'ret3d_log':{'description':'return per day with respect to 3 day before.'}, \
            'ret7d_log':{'description':'return per day with respect to 7 day before.'}, \
            'ret14d_log':{'description':'return per day with respect to 14 day before.'}, \
            'ret30d_log':{'description':'return per day with respect to 30 days before.'}, \
            'ret60d_log':{'description':'return per day with respect to 60 days before.'}, \
            'ret90d_log':{'description':'return per day with respect to 90 days before.'}, \
            'ret120d_log':{'description':'return per day with respect to 120 days before.'}, \
            'ret300d_log':{'description':'return per day with respect to 300 days before.'}, \
            'lastclose_log':{'description':'last-to-close ratio log.'}, \
            'buy_queue_locked':{'description':'indicates whether the trades are in buying queue.'}, \
            'sell_queue_locked':{'description':'indicates whether the trades are in selling queue.'}, \
            }

    return feature_data, feature_detail

def proportion_based_feature(stock_daily_data):
    print('proportion based features')
    feature_data = pd.DataFrame([])
    feature_detail = dict()

    # grouping
    stock_groups = stock_daily_data.groupby('stock_name')

    # map function
    def prp_based_func(stock_group):
        initial_index = stock_group.index
        df = stock_group
        df.index = [pd.Timestamp(item) for item in df['date']]
        return_df = pd.DataFrame({\
            'prp_high30d':df['adj_close_price']/df['adj_max_price'].rolling('30d').max(), \
            'prp_high60d':df['adj_close_price']/df['adj_max_price'].rolling('60d').max(), \
            'prp_high90d':df['adj_close_price']/df['adj_max_price'].rolling('90d').max(), \
            'prp_high120d':df['adj_close_price']/df['adj_max_price'].rolling('120d').max(), \
            'prp_high300d':df['adj_close_price']/df['adj_max_price'].rolling('300d').max(), \
            'prp_low30d':df['adj_close_price']/df['adj_min_price'].rolling('30d').min(), \
            'prp_low60d':df['adj_close_price']/df['adj_min_price'].rolling('60d').min(), \
            'prp_low90d':df['adj_close_price']/df['adj_min_price'].rolling('90d').min(), \
            'prp_low120d':df['adj_close_price']/df['adj_min_price'].rolling('120d').min(), \
            'prp_low300d':df['adj_close_price']/df['adj_min_price'].rolling('300d').min(), \
            'prp_value3d30d':df['value'].rolling('3d').mean()/df['value'].rolling('30d').mean(), \
            'prp_value5d60d':df['value'].rolling('5d').mean()/df['value'].rolling('60d').mean(), \
            'prp_value15d120d':df['value'].rolling('15d').mean()/df['value'].rolling('120d').mean(), \
            'prp_value30d200d':df['value'].rolling('30d').mean()/df['value'].rolling('200d').mean(), \
            })
        return_df.index = initial_index
        return return_df
    
    feature_data = stock_groups.apply(prp_based_func).reset_index(drop=True)
    feature_detail = {\
            'prp_high30d':{'description':'ratio of close_price to maximum of high prices in 30-day period.'}, \
            'prp_high60d':{'description':'ratio of close_price to maximum of high prices in 60-day period.'}, \
            'prp_high90d':{'description':'ratio of close_price to maximum of high prices in 90-day period.'}, \
            'prp_high120d':{'description':'ratio of close_price to maximum of high prices in 120-day period.'}, \
            'prp_high300d':{'description':'ratio of close_price to maximum of high prices in 300-day period.'}, \
            'prp_low30d':{'description':'ratio of close_price to minimum of low prices in 30-day period.'}, \
            'prp_low60d':{'description':'ratio of close_price to minimum of low prices in 60-day period.'}, \
            'prp_low90d':{'description':'ratio of close_price to minimum of low prices in 90-day period.'}, \
            'prp_low120d':{'description':'ratio of close_price to minimum of low prices in 120-day period.'}, \
            'prp_low300d':{'description':'ratio of close_price to minimum of low prices in 300-day period.'}, \
            'prp_value3d30d':{'description':'ratio of 3-day average value to 30-day average value'}, \
            'prp_value5d60d':{'description':'ratio of 5-day average value to 60-day average value'}, \
            'prp_value15d120d':{'description':'ratio of 15-day average value to 120-day average value'}, \
            'prp_value30d200d':{'description':'ratio of 30-day average value to 200-day average value'}, \
            }

    return feature_data, feature_detail

def candle_based_feature(stock_daily_data):

    print('weekly candles features')
    feature_groups = stock_daily_data.groupby('stock_name')

    def candle_based_function(feature_group):
        df = feature_group
        initial_index = feature_group.index
        df = feature_group
        df.index = [pd.Timestamp(item) for item in df['date']]
        return_df = pd.DataFrame({\
            'max_price7d': df['adj_max_price'].rolling(5).max(),\
            'min_price7d': df['adj_min_price'].rolling(5).min(),\
            'first_price7d': df['adj_first_price'].rolling(5).agg(lambda rows: rows[0]),\
            'last_price7d': df['adj_last_price'].rolling(5).agg(lambda rows: rows[-1]),\
            'yesterday_price7d': df['yesterday_price'].rolling(5).agg(lambda rows: rows[0]),\
            'max_price14d': df['adj_max_price'].rolling(10).max(),\
            'min_price14d': df['adj_min_price'].rolling(10).min(),\
            'first_price14d': df['adj_first_price'].rolling(10).agg(lambda rows: rows[0]),\
            'last_price14d': df['adj_last_price'].rolling(10).agg(lambda rows: rows[-1]),\
            'yesterday_price14d': df['yesterday_price'].rolling(10).agg(lambda rows: rows[0]),\
            'max_price21d': df['adj_max_price'].rolling(15).max(),\
            'min_price21d': df['adj_min_price'].rolling(15).min(),\
            'first_price21d': df['adj_first_price'].rolling(15).agg(lambda rows: rows[0]),\
            'last_price21d': df['adj_last_price'].rolling(15).agg(lambda rows: rows[-1]),\
            'yesterday_price21d': df['yesterday_price'].rolling(15).agg(lambda rows: rows[0]),\
            'max_price30d': df['adj_max_price'].rolling(23).max(),\
            'min_price30d': df['adj_min_price'].rolling(23).min(),\
            'first_price30d': df['adj_first_price'].rolling(23).agg(lambda rows: rows[0]),\
            'last_price30d': df['adj_last_price'].rolling(23).agg(lambda rows: rows[-1]),\
            'yesterday_price30d': df['yesterday_price'].rolling(23).agg(lambda rows: rows[0]),\
            'max_price60d': df['adj_max_price'].rolling(44).max(),\
            'min_price60d': df['adj_min_price'].rolling(44).min(),\
            'first_price60d': df['adj_first_price'].rolling(44).agg(lambda rows: rows[0]),\
            'last_price60d': df['adj_last_price'].rolling(44).agg(lambda rows: rows[-1]),\
            'yesterday_price60d': df['yesterday_price'].rolling(44).agg(lambda rows: rows[0]),\
            'max_price90d': df['adj_max_price'].rolling(70).max(),\
            'min_price90d': df['adj_min_price'].rolling(70).min(),\
            'first_price90d': df['adj_first_price'].rolling(70).agg(lambda rows: rows[0]),\
            'last_price90d': df['adj_last_price'].rolling(70).agg(lambda rows: rows[-1]),\
            'yesterday_price90d': df['yesterday_price'].rolling(70).agg(lambda rows: rows[0]),\
            })
        return_df.index = initial_index
        return return_df

    candle_based_data = feature_groups.apply(candle_based_function).reset_index(drop=True)

    candle_based_detail = {\
            'max_price7d': {'description':'max price in 7-day-candle.'},\
            'min_price7d': {'description':'min price in 7-day-candle.'},\
            'first_price7d': {'description':'first price in 7-day-candle.'},\
            'last_price7d': {'description':'last price in 7-day-candle.'},\
            'yesterday_price14d': {'description':'yesterday price in 14-day-candle.'},\
            'max_price14d': {'description':'max price in 14-day-candle.'},\
            'min_price14d': {'description':'min price in 14-day-candle.'},\
            'first_price14d': {'description':'first price in 14-day-candle.'},\
            'last_price14d': {'description':'last price in 14-day-candle.'},\
            'yesterday_price14d': {'description':'yesterday price in 14-day-candle.'},\
            'max_price21d': {'description':'max price in 21-day-candle.'},\
            'min_price21d': {'description':'min price in 21-day-candle.'},\
            'first_price21d': {'description':'first price in 21-day-candle.'},\
            'last_price21d': {'description':'last price in 21-day-candle.'},\
            'yesterday_price21d': {'description':'yesterday price in 21-day-candle.'},\
            'max_price30d': {'description':'max price in 30-day-candle.'},\
            'min_price30d': {'description':'min price in 30-day-candle.'},\
            'first_price30d': {'description':'first price in 30-day-candle.'},\
            'last_price30d': {'description':'last price in 30-day-candle.'},\
            'yesterday_price30d': {'description':'yesterday price in 30-day-candle.'},\
            'max_price60d': {'description':'max price in 60-day-candle.'},\
            'min_price60d': {'description':'min price in 60-day-candle.'},\
            'first_price60d': {'description':'first price in 60-day-candle.'},\
            'last_price60d': {'description':'last price in 60-day-candle.'},\
            'yesterday_price60d': {'description':'yesterday price in 60-day-candle.'},\
            'max_price90d': {'description':'max price in 90-day-candle.'},\
            'min_price90d': {'description':'min price in 90-day-candle.'},\
            'first_price90d': {'description':'first price in 90-day-candle.'},\
            'last_price90d': {'description':'last price in 90-day-candle.'},\
            'yesterday_price90d': {'description':'yesterday price in 90-day-candle.'},\
            }

    price_data = pd.concat([stock_daily_data,candle_based_data],axis=1)

    feature_groups = price_data.groupby('stock_name')

    def shadow_based_featurs(feature_group):
        df = feature_group
        initial_index = feature_group.index
        df = feature_group
        df.index = [pd.Timestamp(item) for item in df['date']]
        
        return_df = pd.DataFrame({\
            'shadow_up7d': (df['max_price7d'] - np.maximum(df['first_price7d'],df['last_price7d']))/
                                (df['yesterday_price7d']),\
            'shadow_low7d': (df['min_price7d'] - np.minimum(df['first_price7d'],df['last_price7d']))/
                                (df['yesterday_price7d']),\
            'body7d': np.abs(df['last_price7d']-df['first_price7d'])/df['yesterday_price7d'],\
            'shadow_up14d': (df['max_price14d'] - np.maximum(df['first_price14d'],df['last_price14d']))/
                                (df['yesterday_price14d']),\
            'shadow_low14d': (df['min_price14d'] - np.minimum(df['first_price14d'],df['last_price14d']))/
                                (df['yesterday_price14d']),\
            'body14d': np.abs(df['last_price14d']-df['first_price14d'])/df['yesterday_price14d'],\
            'shadow_up21d': (df['max_price21d'] - np.maximum(df['first_price21d'],df['last_price21d']))/
                                (df['yesterday_price21d']),\
            'shadow_low21d': (df['min_price21d'] - np.minimum(df['first_price21d'],df['last_price21d']))/
                                (df['yesterday_price21d']),\
            'body21d': np.abs(df['last_price21d']-df['first_price21d'])/df['yesterday_price21d'],\
            'shadow_up30d': (df['max_price30d'] - np.maximum(df['first_price30d'],df['last_price30d']))/
                                (df['yesterday_price30d']),\
            'shadow_low30d': (df['min_price30d'] - np.minimum(df['first_price30d'],df['last_price30d']))/
                                (df['yesterday_price30d']),\
            'body30d': np.abs(df['last_price30d']-df['first_price30d'])/df['yesterday_price30d'],\
            'shadow_up60d': (df['max_price60d'] - np.maximum(df['first_price60d'],df['last_price60d']))/
                                (df['yesterday_price60d']),\
            'shadow_low60d': (df['min_price60d'] - np.minimum(df['first_price60d'],df['last_price60d']))/
                                (df['yesterday_price60d']),\
            'body60d': np.abs(df['last_price60d']-df['first_price60d'])/df['yesterday_price60d'],\
            'shadow_up90d': (df['max_price90d'] - np.maximum(df['first_price90d'],df['last_price90d']))/
                                (df['yesterday_price90d']),\
            'shadow_low90d': (df['min_price90d'] - np.minimum(df['first_price90d'],df['last_price90d']))/
                                (df['yesterday_price90d']),\
            'body90d': np.abs(df['last_price90d']-df['first_price90d'])/df['yesterday_price90d']\
            })
        return_df.index = initial_index
        return return_df

    shadow_based_data = feature_groups.apply(shadow_based_featurs).reset_index(drop=True) 

    shadow_based_detail = {\
            'shadow_up7d': {'description':'max price in 7-day-candle.'},\
            'shadow_low7d': {'description':'min price in 7-day-candle.'},\
            'body7d': {'description':'first price in 7-day-candle.'},\
            'shadow_up14d': {'description':'max price in 14-day-candle.'},\
            'shadow_low14d': {'description':'min price in 14-day-candle.'},\
            'body14d': {'description':'first price in 14-day-candle.'},\
            'shadow_up21d': {'description':'max price in 21-day-candle.'},\
            'shadow_low21d': {'description':'min price in 21-day-candle.'},\
            'body21d': {'description':'first price in 21-day-candle.'},\
            'shadow_up30d': {'description':'max price in 30-day-candle.'},\
            'shadow_low30d': {'description':'min price in 30-day-candle.'},\
            'body30d': {'description':'first price in 30-day-candle.'},\
            'shadow_up60d': {'description':'max price in 60-day-candle.'},\
            'shadow_low60d': {'description':'min price in 60-day-candle.'},\
            'body60d': {'description':'first price in 60-day-candle.'},\
            'shadow_up90d': {'description':'max price in 90-day-candle.'},\
            'shadow_low90d': {'description':'min price in 90-day-candle.'},\
            'body90d': {'description':'first price in 90-day-candle.'}\
    }
    

    feature_data = pd.concat([shadow_based_data, candle_based_data ], \
        axis=1)
    feature_detail = dict()
    for d in [candle_based_detail, shadow_based_detail]:
        feature_detail.update(d)

    return feature_data, feature_detail

def indv_nonindv_based_feature(stock_daily_data, wght_data, prp_data):
    print('individual and non-individual features')
    feature_data = pd.DataFrame([])
    feature_detail = dict()

    # grouping
    stock_groups = stock_daily_data.groupby('stock_name')

    # per capita map function
    def pcap_based_func(stock_group):
        initial_index = stock_group.index
        df = stock_group
        df.index = [pd.Timestamp(item) for item in df['date']]
        return_df = pd.DataFrame({\
            'indv_buy_pcap':np.where(df['Individual_buy_count'] <= 0, \
                0.0, df['Individual_buy_value']/df['Individual_buy_count']), \
            'indv_sell_pcap':np.where(df['Individual_sell_count'] <= 0, \
                0.0, df['Individual_sell_value']/df['Individual_sell_count']), \
            'indv_net_count_prp7d30':(df['Individual_buy_count']-df['Individual_sell_count']).rolling('7d').sum()/\
                (df['Individual_buy_count']+df['Individual_sell_count']).rolling('30d').sum(), \
            'indv_net_count_prp7d120':(df['Individual_buy_count']-df['Individual_sell_count']).rolling('7d').sum()/\
                (df['Individual_buy_count']+df['Individual_sell_count']).rolling('120d').sum(), \
            'indv_net_value':(df['Individual_buy_value']-df['Individual_sell_value']), \
            'indv_relnet_value_d14':((df['Individual_buy_value']-df['Individual_sell_value'])/\
                (df['Individual_buy_value']+df['Individual_sell_value'])).rolling('14d').sum(), \
            'indv_relnet_value_d30':((df['Individual_buy_value']-df['Individual_sell_value'])/\
                (df['Individual_buy_value']+df['Individual_sell_value'])).rolling('30d').sum(), \
            'indv_relnet_value_d60':((df['Individual_buy_value']-df['Individual_sell_value'])/\
                (df['Individual_buy_value']+df['Individual_sell_value'])).rolling('60d').sum(), \
            'indv_relnet_value_d90':((df['Individual_buy_value']-df['Individual_sell_value'])/\
                (df['Individual_buy_value']+df['Individual_sell_value'])).rolling('90d').sum(), \
            'indv_relnet_value_d120':((df['Individual_buy_value']-df['Individual_sell_value'])/\
                (df['Individual_buy_value']+df['Individual_sell_value'])).rolling('120d').sum(), \
            'indv_relnet_value_d300':((df['Individual_buy_value']-df['Individual_sell_value'])/\
                (df['Individual_buy_value']+df['Individual_sell_value'])).rolling('300d').sum(), \
            'indv_relnet5d10d_value_d10':((df['Individual_buy_value'].rolling('5d').mean()-df['Individual_sell_value'].rolling('10d').mean())/\
                (df['Individual_buy_value']+df['Individual_sell_value'])).rolling('10d').mean(), \
            'nonindv_relnet5d10d_value_d10':((df['NonIndividual_buy_value'].rolling('5d').mean()-df['NonIndividual_sell_value'].rolling('10d').mean())/\
                (df['NonIndividual_buy_value']+df['NonIndividual_sell_value'])).rolling('10d').mean(), \
            'nonindv_buy_pcap':np.where(df['NonIndividual_buy_count'] <= 0, \
                0.0, df['NonIndividual_buy_value']/df['NonIndividual_buy_count']), \
            'nonindv_sell_pcap':np.where(df['NonIndividual_sell_count'] <= 0, \
                0.0, df['NonIndividual_sell_value']/df['NonIndividual_sell_count']), \
            'nonindv_net_count_prp7d30':(df['NonIndividual_buy_count']-df['NonIndividual_sell_count']).rolling('7d').sum()/\
                (df['NonIndividual_buy_count']+df['NonIndividual_sell_count']).rolling('30d').sum(), \
            'nonindv_net_count_prp7d120':(df['NonIndividual_buy_count']-df['NonIndividual_sell_count']).rolling('7d').sum()/\
                (df['NonIndividual_buy_count']+df['NonIndividual_sell_count']).rolling('120d').sum(), \
            'nonindv_net_value':(df['NonIndividual_buy_value']-df['NonIndividual_sell_value']), \
            'nonindv_relnet_value_d14':((df['NonIndividual_buy_value']-df['NonIndividual_sell_value'])/\
                (df['NonIndividual_buy_value']+df['NonIndividual_sell_value'])).rolling('14d').sum(), \
            'nonindv_relnet_value_d30':((df['NonIndividual_buy_value']-df['NonIndividual_sell_value'])/\
                (df['NonIndividual_buy_value']+df['NonIndividual_sell_value'])).rolling('30d').sum(), \
            'nonindv_relnet_value_d60':((df['NonIndividual_buy_value']-df['NonIndividual_sell_value'])/\
                (df['NonIndividual_buy_value']+df['NonIndividual_sell_value'])).rolling('60d').sum(), \
            'nonindv_relnet_value_d90':((df['NonIndividual_buy_value']-df['NonIndividual_sell_value'])/\
                (df['NonIndividual_buy_value']+df['NonIndividual_sell_value'])).rolling('90d').sum(), \
            'nonindv_relnet_value_d120':((df['NonIndividual_buy_value']-df['NonIndividual_sell_value'])/\
                (df['NonIndividual_buy_value']+df['NonIndividual_sell_value'])).rolling('120d').sum(), \
            'nonindv_relnet_value_d300':((df['NonIndividual_buy_value']-df['NonIndividual_sell_value'])/\
                (df['NonIndividual_buy_value']+df['NonIndividual_sell_value'])).rolling('300d').sum(), \
            'indv_buy_ratio':df['Individual_buy_value']/(df['Individual_buy_value'] + df['NonIndividual_buy_value']), \
            'nonindv_sell_ratio':df['NonIndividual_sell_value']/(df['Individual_sell_value'] + df['NonIndividual_sell_value']), \
            })
        return_df.index = initial_index
        return return_df
    
    pcap_feature_data = stock_groups.apply(pcap_based_func).reset_index(drop=True)
    pcap_feature_detail = {\
            'indv_buy_pcap':{'description':'buy value per capita for indivitual clients.'}, \
            'indv_sell_pcap':{'description':'sell value per capita for indivitual clients.'}, \
            'indv_net_count_prp7d30':{'description':'ratio of 7-day difference of buy & sell count to 30-day sum of buy & sell count  for indivitual clients.'}, \
            'indv_net_count_prp7d120':{'description':'ratio of 7-day difference of buy & sell count to 120-day sum of buy & sell count  for indivitual clients.'}, \
            'indv_net_value':{'description':'difference of buy & sell value  for indivitual clients.'}, \
            'indv_relnet_value_d14':{'description':'14-day mean ratio of difference of buy & sell value to sum of buy & sell value  for indivitual clients.'}, \
            'indv_relnet_value_d30':{'description':'30-day mean ratio of difference of buy & sell value to sum of buy & sell value  for indivitual clients.'}, \
            'indv_relnet_value_d60':{'description':'60-day mean ratio of difference of buy & sell value to sum of buy & sell value  for indivitual clients.'}, \
            'indv_relnet_value_d90':{'description':'90-day mean ratio of difference of buy & sell value to sum of buy & sell value  for indivitual clients.'}, \
            'indv_relnet_value_d120':{'description':'120-day mean ratio of difference of buy & sell value to sum of buy & sell value  for indivitual clients.'}, \
            'indv_relnet_value_d300':{'description':'300-day mean ratio of difference of buy & sell value to sum of buy & sell value  for indivitual clients.'}, \
            'indv_relnet5d10d_value_d10':{'description':'ratio of dev of 10-day and 5-day net value to the 10day value mean for individual clients',
                                          'level_agg': True}, \
            'nonindv_relnet5d10d_value_d10':{'description':'ratio of dev of 10-day and 5-day net value to the 10day value mean for nonindividual clients',
                                             'level_agg': True},\
            'nonindv_buy_pcap':{'description':'buy value per capita for non-indivitual clients.'}, \
            'nonindv_sell_pcap':{'description':'sell value per capita for non-indivitual clients.'}, \
            'nonindv_net_count_prp7d30':{'description':'ratio of 7-day difference of buy & sell count to 30-day sum of buy & sell count  for non-indivitual clients.'}, \
            'nonindv_net_count_prp7d120':{'description':'ratio of 7-day difference of buy & sell count to 120-day sum of buy & sell count  for non-indivitual clients.'}, \
            'nonindv_net_value':{'description':'difference of buy & sell value  for non-indivitual clients.'}, \
            'nonindv_relnet_value_d14':{'description':'14-day mean ratio of difference of buy & sell value to sum of buy & sell value  for non-indivitual clients.'}, \
            'nonindv_relnet_value_d30':{'description':'30-day mean ratio of difference of buy & sell value to sum of buy & sell value  fo non-indivitual clients.'}, \
            'nonindv_relnet_value_d60':{'description':'60-day mean ratio of difference of buy & sell value to sum of buy & sell value  fo non-indivitual clients.'}, \
            'nonindv_relnet_value_d90':{'description':'90-day mean ratio of difference of buy & sell value to sum of buy & sell value  for non-indivitual clients.'}, \
            'nonindv_relnet_value_d120':{'description':'120-day mean ratio of difference of buy & sell value to sum of buy & sell value  for non-indivitual clients.'}, \
            'nonindv_relnet_value_d300':{'description':'300-day mean ratio of difference of buy & sell value to sum of buy & sell value  for non-indivitual clients.'}, \
            'indv_buy_ratio':{'description':'ratio of buy for individual clients to buy for all clients.'}, \
            'nonindv_sell_ratio':{'description':'ratio of sell for non-individual clients to sell for all clients.'}, \
            }

    # stock percapita dataframe 
    stock_percapita = pd.concat([stock_daily_data[['date', 'stock_name', 'Individual_buy_value', 'Individual_sell_value', 'NonIndividual_buy_value', 'NonIndividual_sell_value']], \
        pcap_feature_data, wght_data, prp_data], axis=1)
    
    # per capita map function
    stock_percapita_date_groups = stock_percapita.groupby('date')
    def mrkt_pcap_based_func(stock_date_group):
        initial_index = stock_date_group.index
        df = stock_date_group
        one_vec = np.ones([len(df)])
        return_df = pd.DataFrame({\
            'mrkt_indv_buy_pcap':(df['indv_buy_pcap']*df['value_weight20d']).sum()*one_vec, \
            'mrkt_indv_sell_pcap':(df['indv_sell_pcap']*df['value_weight20d']).sum()*one_vec, \
            'mrkt_indv_net_value':((df['Individual_buy_value'] - df['Individual_sell_value'])*df['value_weight20d']).sum()*one_vec, \
            'mrkt_indv_absnet_value':(np.abs(df['Individual_buy_value'] - df['Individual_sell_value'])*df['value_weight20d']).sum()*one_vec, \
            'mrkt_nonindv_buy_pcap':(df['nonindv_buy_pcap']*df['value_weight20d']).sum()*one_vec, \
            'mrkt_nonindv_sell_pcap':(df['nonindv_sell_pcap']*df['value_weight20d']).sum()*one_vec, \
            'mrkt_nonindv_net_value':((df['NonIndividual_buy_value'] - df['NonIndividual_sell_value'])*df['value_weight20d']).sum()*one_vec, \
            'mrkt_nonindv_absnet_value':(np.abs(df['NonIndividual_buy_value'] - df['NonIndividual_sell_value'])*df['value_weight20d']).sum()*one_vec, \
            'mrkt_prp_high30d':df['prp_high30d'].mean()*one_vec,\
            'mrkt_prp_high60d':df['prp_high60d'].mean()*one_vec,\
            'mrkt_prp_high90d':df['prp_high90d'].mean()*one_vec,\
            'mrkt_prp_high120d':df['prp_high120d'].mean()*one_vec,\
            'mrkt_prp_high300d':df['prp_high300d'].mean()*one_vec,\
            'mrkt_prp_low30d':df['prp_low30d'].mean()*one_vec,\
            'mrkt_prp_low60d':df['prp_low60d'].mean()*one_vec,\
            'mrkt_prp_low90d':df['prp_low90d'].mean()*one_vec,\
            'mrkt_prp_low120d':df['prp_low120d'].mean()*one_vec,\
            'mrkt_prp_low300d':df['prp_low300d'].mean()*one_vec\
            })
        return_df.index = initial_index
        return return_df
    
    mrkt_pcap_feature_data = stock_percapita_date_groups.apply(mrkt_pcap_based_func).reset_index(drop=True)
    mrkt_pcap_feature_detail = {\
            'mrkt_indv_buy_pcap':{'description':'individual buy per capita for daily market.'}, \
            'mrkt_indv_sell_pcap':{'description':'individual sell per capita for daily market.'}, \
            'mrkt_indv_net_value':{'description':'individual net value (buy sell difference) for daily market.'}, \
            'mrkt_indv_absnet_value':{'description':'individual absolute net value (buy sell difference) for daily market.'}, \
            'mrkt_nonindv_buy_pcap':{'description':'non-individual buy per capita for daily market.'}, \
            'mrkt_nonindv_sell_pcap':{'description':'non-individual sell per capita for daily market.'}, \
            'mrkt_nonindv_net_value':{'description':'non-individual net value (buy sell difference) for daily market.'}, \
            'mrkt_nonindv_absnet_value':{'description':'non-individual absolute net value (buy sell difference) for daily market.'}, \
            'mrkt_prp_high30d': {'description':'mean of prp_high30d in all market per day.'},\
            'mrkt_prp_high60d': {'description':'mean of prp_high60d in all market per day.'},\
            'mrkt_prp_high90d': {'description':'mean of prp_high90d in all market per day.'},\
            'mrkt_prp_high120d':{'description':'mean of prp_high120d in all market per day.'},\
            'mrkt_prp_high300d':{'description':'mean of prp_high300d in all market per day.'},\
            'mrkt_prp_low30d': {'description':'mean of prp_low30d in all market per day.'},\
            'mrkt_prp_low60d': {'description':'mean of prp_low60d in all market per day.'},\
            'mrkt_prp_low90d': {'description':'mean of prp_low90d in all market per day.'},\
            'mrkt_prp_low120d': {'description':'mean of prp_low120d in all market per day.'},\
            'mrkt_prp_low300d': {'description':'mean of prp_low300d in all market per day.'}\
            }
    # stock percapita dataframe 
    stock_percapita = pd.concat([stock_percapita, mrkt_pcap_feature_data], axis=1)

    # market rolling base
    stock_percapita_groups = stock_percapita.groupby('stock_name')
    def mrkt_rolling_based_func(stock_percapita_group):
        initial_index = stock_percapita_group.index
        df = stock_percapita_group
        df.index = [pd.Timestamp(item) for item in df['date']]
        return_df = pd.DataFrame({\
        'mrkt_indv_absnet_value3d': df['mrkt_indv_absnet_value'].rolling('3d').sum(),\
        'mrkt_nonindv_absnet_value3d': df['mrkt_nonindv_absnet_value'].rolling('3d').sum()\
        })
        return_df.index = initial_index
        return return_df
        
    mrkt_rolling_data = stock_percapita_groups.apply(mrkt_rolling_based_func).reset_index(drop=True)
    mrkt_rolling_detail = {\
        'mrkt_indv_absnet_value3d': {'description': 'individual absolute net value (buy sell difference) for daily market in recent 3 days.'},\
        'mrkt_nonindv_absnet_value3d': {'description':'nonindividual absolute net value (buy sell difference) for daily market in recent 3 days.'}\
    }

    prc_mrkt_data = pd.concat([stock_percapita,mrkt_rolling_data],axis=1)

    # per capita map function
    # TODO: for daily powers => make them bolder!/ for others: use mean instead of sum
    prc_mrkt_data_groups = prc_mrkt_data.groupby('stock_name')
    def relpcap_based_func(prc_mrkt_data_group):
        initial_index = prc_mrkt_data_group.index
        df = prc_mrkt_data_group
        df.index = [pd.Timestamp(item) for item in df['date']]
        return_df = pd.DataFrame({\
            'indv_power':((df['indv_buy_pcap'] - df['indv_sell_pcap'])\
                /(df['indv_buy_pcap'] + df['indv_sell_pcap'])), \
            'nonindv_power':(df['nonindv_buy_pcap'] - df['nonindv_sell_pcap'])\
                /(df['nonindv_buy_pcap'] + df['nonindv_sell_pcap']), \
            'indv_power3d':(df['indv_buy_pcap'] - df['indv_sell_pcap']).rolling('3d').sum()\
                /(df['indv_buy_pcap'] + df['indv_sell_pcap']).rolling('3d').mean(), \
            'nonindv_power3d':(df['nonindv_buy_pcap'] - df['nonindv_sell_pcap']).rolling('3d').sum()\
                /(df['nonindv_buy_pcap'] + df['nonindv_sell_pcap']).rolling('3d').mean(),\
            'indv_power5d':(df['indv_buy_pcap'] - df['indv_sell_pcap']).rolling('5d').sum()\
                /(df['indv_buy_pcap'] + df['indv_sell_pcap']).rolling('5d').mean(), \
            'nonindv_power5d':(df['nonindv_buy_pcap'] - df['nonindv_sell_pcap']).rolling('5d').sum()\
                /(df['nonindv_buy_pcap'] + df['nonindv_sell_pcap']).rolling('5d').mean(),\
            'indv_power7d':(df['indv_buy_pcap'] - df['indv_sell_pcap']).rolling('7d').sum()\
                /(df['indv_buy_pcap'] + df['indv_sell_pcap']).rolling('7d').mean(), \
            'nonindv_power7d':(df['nonindv_buy_pcap'] - df['nonindv_sell_pcap']).rolling('7d').sum()\
                /(df['nonindv_buy_pcap'] + df['nonindv_sell_pcap']).rolling('7d').mean(), \
            'indv_power14d':(df['indv_buy_pcap'] - df['indv_sell_pcap']).rolling('14d').sum()\
                /(df['indv_buy_pcap'] + df['indv_sell_pcap']).rolling('14d').mean(), \
            'nonindv_power14d':(df['nonindv_buy_pcap'] - df['nonindv_sell_pcap']).rolling('14d').sum()\
                /(df['nonindv_buy_pcap'] + df['nonindv_sell_pcap']).rolling('14d').mean(),\
            'indv_power30d':(df['indv_buy_pcap'] - df['indv_sell_pcap']).rolling('30d').sum()\
                /(df['indv_buy_pcap'] + df['indv_sell_pcap']).rolling('30d').mean(), \
            'nonindv_power30d':(df['nonindv_buy_pcap'] - df['nonindv_sell_pcap']).rolling('30d').sum()\
                /(df['nonindv_buy_pcap'] + df['nonindv_sell_pcap']).rolling('30d').mean(), \
            'indv_power5d25':(df['indv_buy_pcap'] - df['indv_sell_pcap']).rolling('5d').sum()\
                /(df['indv_buy_pcap'] - df['indv_sell_pcap']).rolling('25d').sum(),\
            'nonindv_power5d25':(df['nonindv_buy_pcap'] - df['nonindv_sell_pcap']).rolling('5d').sum()\
                /(df['nonindv_buy_pcap'] - df['nonindv_sell_pcap']).rolling('25d').sum(),\
            'indv_buy_pcap_prpd7d30':df['indv_buy_pcap'].rolling('7d').mean()/df['indv_buy_pcap'].rolling('30d').mean(), \
            'indv_buy_pcap_prpd14d60':df['indv_buy_pcap'].rolling('14d').mean()/df['indv_buy_pcap'].rolling('60d').mean(), \
            'indv_buy_pcap_prpd30d120':df['indv_buy_pcap'].rolling('30d').mean()/df['indv_buy_pcap'].rolling('120d').mean(), \
            'indv_sell_pcap_prpd7d30':df['indv_sell_pcap'].rolling('7d').mean()/df['indv_sell_pcap'].rolling('30d').mean(), \
            'indv_sell_pcap_prpd14d60':df['indv_sell_pcap'].rolling('14d').mean()/df['indv_sell_pcap'].rolling('60d').mean(), \
            'indv_sell_pcap_prpd30d120':df['indv_sell_pcap'].rolling('30d').mean()/df['indv_sell_pcap'].rolling('120d').mean(), \
            'nonindv_buy_pcap_prpd7d30':df['nonindv_buy_pcap'].rolling('7d').mean()/df['nonindv_buy_pcap'].rolling('30d').mean(), \
            'nonindv_buy_pcap_prpd14d60':df['nonindv_buy_pcap'].rolling('14d').mean()/df['nonindv_buy_pcap'].rolling('60d').mean(), \
            'nonindv_buy_pcap_prpd30d120':df['nonindv_buy_pcap'].rolling('30d').mean()/df['nonindv_buy_pcap'].rolling('120d').mean(), \
            'nonindv_sell_pcap_prpd7d30':df['nonindv_sell_pcap'].rolling('7d').mean()/df['nonindv_sell_pcap'].rolling('30d').mean(), \
            'nonindv_sell_pcap_prpd14d60':df['nonindv_sell_pcap'].rolling('14d').mean()/df['nonindv_sell_pcap'].rolling('60d').mean(), \
            'nonindv_sell_pcap_prpd30d120':df['nonindv_sell_pcap'].rolling('30d').mean()/df['nonindv_sell_pcap'].rolling('120d').mean(), \
            'indv_buy_pcap_prp_mrkt':df['indv_buy_pcap']/df['mrkt_indv_buy_pcap'], \
            'indv_sell_pcap_prp_mrkt':df['indv_sell_pcap']/df['mrkt_indv_sell_pcap'], \
            'nonindv_buy_pcap_prp_mrkt':df['nonindv_buy_pcap']/df['mrkt_nonindv_buy_pcap'], \
            'nonindv_sell_pcap_prp_mrkt':df['nonindv_sell_pcap']/df['mrkt_nonindv_sell_pcap'], \
            'indv_net_dev_mrkt':(df['indv_net_value'] - df['mrkt_indv_net_value'])/df['mrkt_indv_absnet_value'], \
            'indv_net_dev_mrkt3d':((df['indv_net_value'] - df['mrkt_indv_net_value']).rolling('3d').sum())/df['mrkt_indv_absnet_value3d'], \
            'nonindv_net_dev_mrkt':(df['nonindv_net_value'] - df['mrkt_nonindv_net_value'])/df['mrkt_nonindv_absnet_value'], \
            'nonindv_net_dev_mrkt3d':((df['nonindv_net_value'] - df['mrkt_nonindv_net_value']).rolling('3d').sum())/df['mrkt_nonindv_absnet_value3d'], \
            'prp_high30d_dev_mrkt':df['prp_high30d'] - df['mrkt_prp_high30d'], \
            'prp_high60d_dev_mrkt':df['prp_high60d'] - df['mrkt_prp_high60d'], \
            'prp_high90d_dev_mrkt':df['prp_high90d'] - df['mrkt_prp_high90d'], \
            'prp_high120d_dev_mrkt':df['prp_high120d'] - df['mrkt_prp_high120d'], \
            'prp_high300d_dev_mrkt':df['prp_high300d'] - df['mrkt_prp_high300d'], \
            'prp_low30d_dev_mrkt':df['prp_low30d'] - df['mrkt_prp_low30d'], \
            'prp_low60d_dev_mrkt':df['prp_low60d'] - df['mrkt_prp_low60d'], \
            'prp_low90d_dev_mrkt':df['prp_low90d'] - df['mrkt_prp_low90d'], \
            'prp_low120d_dev_mrkt':df['prp_low120d'] - df['mrkt_prp_low120d'], \
            'prp_low300d_dev_mrkt':df['prp_low300d'] - df['mrkt_prp_low300d'], \
            })
        return_df.index = initial_index
        return return_df
    
    relpcap_feature_data = prc_mrkt_data_groups.apply(relpcap_based_func).reset_index(drop=True)
    relpcap_feature_detail = {\
            'indv_power':{'description':'buy to sell ratio for individual clients'}, \
            'nonindv_power':{'description':'buy to sell ratio for non-individual clients'}, \
            'indv_power3d':{'description':'buy to sell ratio for individual clients for last 3 days'}, \
            'nonindv_power3d':{'description':'buy to sell ratio for non-individual clients for last 3 days'}, \
            'indv_power5d':{'description':'buy to sell ratio for non-individual clients for last 5 days'},\
            'nonindv_power5d':{'description':'buy to sell ratio for non-individual clients for last 5 days'},\
            'indv_power7d':{'description':'buy to sell ratio for individual clients for last 7 days'}, \
            'nonindv_powerd7':{'description':'buy to sell ratio for non-individual clients for last 7 days'}, \
            'indv_power14d':{'description':'buy to sell ratio for individual clients for last 14 days'}, \
            'nonindv_power14d':{'description':'buy to sell ratio for non-individual clients for last 14 days'}, \
            'indv_power30d':{'description':'buy to sell ratio for individual clients for last 30 days'}, \
            'nonindv_power30d':{'description':'buy to sell ratio for non-individual clients for last 30 days',
                                'level_agg': True}, \
            'indv_power5d25':{'desciption':'ratio of buy-sell deviation in 5 days to 25days for individual client',
                                'level_agg': True},\
            'nonindv_power5d25':{'desciption':'ratio of buy-sell deviation in 5 days to 25days for nonindividual client'},\
            'indv_buy_pcap_prpd7d30':{'description':'ratio of buy in last 7 days to buy in last 30 days for individual clients'}, \
            'indv_buy_pcap_prpd14d60':{'description':'ratio of buy in last 14 days to buy in last 60 days for individual clients'}, \
            'indv_buy_pcap_prpd30d120':{'description':'ratio of buy in last 30 days to buy in last 120 days for individual clients'}, \
            'indv_sell_pcap_prpd7d30':{'description':'ratio of sell in last 7 days to sell in last 30 days for individual clients'}, \
            'indv_sell_pcap_prpd14d60':{'description':'ratio of sell in last 14 days to sell in last 60 days for individual clients'}, \
            'indv_sell_pcap_prpd30d120':{'description':'ratio of sell in last 30 days to sell in last 120 days for individual clients'}, \
            'nonindv_buy_pcap_prpd7d30':{'description':'ratio of buy in last 7 days to buy in last 30 days for non-individual clients'}, \
            'nonindv_buy_pcap_prpd14d60':{'description':'ratio of buy in last 14 days to buy in last 60 days for non-individual clients'}, \
            'nonindv_buy_pcap_prpd30d120':{'description':'ratio of buy in last 30 days to buy in last 120 days for non-individual clients'}, \
            'nonindv_sell_pcap_prpd7d30':{'description':'ratio of sell in last 7 days to sell in last 30 days for non-individual clients'}, \
            'nonindv_sell_pcap_prpd14d60':{'description':'ratio of sell in last 14 days to sell in last 60 days for non-individual clients'}, \
            'nonindv_sell_pcap_prpd30d120':{'description':'ratio of sell in last 30 days to sell in last 120 days for non-individual clients'}, \
            'indv_buy_pcap_prp_mrkt':{'description':'ratio of buy per capita to weighted average of buy percapita of all market for individual clients'}, \
            'indv_sell_pcap_prp_mrkt':{'description':'ratio of sell per capita to weighted average of sell percapita of all market for individual clients'}, \
            'nonindv_buy_pcap_prp_mrkt':{'description':'ratio of buy per capita to weighted average of buy percapita of all market for non-individual clients'}, \
            'nonindv_sell_pcap_prp_mrkt':{'description':'ratio of sell per capita to weighted average of sell percapita of all market for non-individual clients'}, \
            'indv_net_dev_mrkt':{'description':'ratio of net-to-market deviation to absolue market net for individual clients'}, \
            'indv_net_dev_mrkt3d':{'description':'ratio of net-to-market deviation to absolue market net for individual clients in 3 days'}, \
            'nonindv_net_dev_mrkt':{'description':'ratio of net-to-market deviation to absolue market net for non-individual clients'}, \
            'nonindv_net_dev_mrkt3d':{'description':'ratio of net-to-market deviation to absolue market net for non-individual clients in 3 days'}, \
            'prp_high30d_dev_mrkt':{'description':'deviation of prp_high30d from all market'}, \
            'prp_high60d_dev_mrkt':{'description':'deviation of prp_high60d from all market'}, \
            'prp_high90d_dev_mrkt':{'description':'deviation of prp_high90d from all market'}, \
            'prp_high120d_dev_mrkt':{'description':'deviation of prp_high120d from all market'}, \
            'prp_high300d_dev_mrkt':{'description':'deviation of prp_high300d from all market'}, \
            'prp_low30d_dev_mrkt':{'description':'deviation of prp_low30d from all market'}, \
            'prp_low60d_dev_mrkt':{'description':'deviation of prp_low60d from all market'}, \
            'prp_low90d_dev_mrkt':{'description':'deviation of prp_low90d from all market'}, \
            'prp_low120d_dev_mrkt':{'description':'deviation of prp_low120d from all market'}, \
            'prp_low300d_dev_mrkt':{'description':'deviation of prp_low300d from all market'}, \
            }


    relpcap_feature_ = pd.concat([stock_percapita,relpcap_feature_data],axis=1)
    relpcap_feature_groups = relpcap_feature_.groupby('stock_name')

    def power_count_features(relpcap_feature_group):
        initial_index = relpcap_feature_group.index
        df = relpcap_feature_group
        df.index = [pd.Timestamp(item) for item in df['date']]

        count_indices = np.where(df['indv_power']>=0)
        temp = np.zeros(len(df['indv_power']),)
        temp[count_indices,] = 1
        df['count_indices'] = temp
        count_5d = df['count_indices'].rolling('5d').sum()
        count_30d = df['count_indices'].rolling('30d').sum()

        return_df = pd.DataFrame({\
            'ind_power_count5d30' : count_5d/count_30d} )
        return_df.index = initial_index
        return return_df

    power_count_data = relpcap_feature_groups.apply(power_count_features).reset_index(drop=True)
    power_count_detail = {\
        'ind_power_count5d30' : {'decsription':'ratio of  non-negative power-count in 5 days to the non-negative power count in 30 days',
                                'level_agg' : True}\
                        }
    # concate the results
    feature_data = pd.concat([pcap_feature_data, mrkt_pcap_feature_data, mrkt_rolling_data, relpcap_feature_data,power_count_data], \
        axis=1)
    feature_detail = dict()
    for d in [pcap_feature_detail, mrkt_pcap_feature_detail, mrkt_rolling_detail, relpcap_feature_detail,power_count_detail]:
        feature_detail.update(d)

    return feature_data, feature_detail

def calender_based_feature(stock_daily_data):
    print('calendar based features')
    feature_data = pd.DataFrame([])
    feature_detail = dict()

    # converge
    jdatetime.set_locale('fa_IR')
    def jcal_func(row):
        geogdate = datetime.datetime.strptime(row['date'], '%Y-%m-%d').date()
        jalalidate = jdatetime.date.fromgregorian(day=geogdate.day,month=geogdate.month,year=geogdate.year)
        return pd.Series([jalalidate.strftime("%Y-%m-%d"), jalalidate.weekday()], index=['jdate', 'jweekday'])

    feature_data = stock_daily_data.swifter.apply(jcal_func, axis=1)
    feature_detail = {\
            'jdate':{'description':'jalali date.'}, \
            'jweekday':{'description':'jalali week day (starting with 0 for Shanbeh).'}, \
            }

    return feature_data, feature_detail

def fwd_prc_based_feature(merged_data_df):
    print('forward price based features')
    feature_data = pd.DataFrame([])
    feature_detail = dict()

    # grouping
    stock_groups = merged_data_df.groupby('stock_name')

    # map function
    def fwdret_based_func(stock_group):
        initial_index = stock_group.index
        df = stock_group
        df.index = [pd.Timestamp(item) for item in df['date']]

        ret_nanfilled_df = (np.log(df['close_price']) - np.log(df['yesterday_price'])).reindex(\
            pd.date_range(df.index.min(), df.index.max()), fill_value="NaN")
        ret_nanfilled_df.name = None
        indret_nanfilled_df = (np.log((df['total_stock_index']).pct_change() + 1)).reindex(\
            pd.date_range(df.index.min(), df.index.max()), fill_value="NaN")
        indret_nanfilled_df.name = None

        full_ret2d = pd.DataFrame(ret_nanfilled_df.rolling('2d').sum().shift(-1))
        full_ret7d = pd.DataFrame(ret_nanfilled_df.rolling('7d').sum().shift(-6))
        full_ret14d = pd.DataFrame(ret_nanfilled_df.rolling('14d').sum().shift(-13))
        full_ret21d = pd.DataFrame(ret_nanfilled_df.rolling('21d').sum().shift(-20))
        full_ret30d = pd.DataFrame(ret_nanfilled_df.rolling('30d').sum().shift(-29))
        full_ret60d = pd.DataFrame(ret_nanfilled_df.rolling('60d').sum().shift(-59))
        full_ret90d = pd.DataFrame(ret_nanfilled_df.rolling('90d').sum().shift(-89))

        full_indret2d = pd.DataFrame(indret_nanfilled_df.rolling('2d').sum().shift(-1))
        full_indret7d = pd.DataFrame(indret_nanfilled_df.rolling('7d').sum().shift(-6))
        full_indret14d = pd.DataFrame(indret_nanfilled_df.rolling('14d').sum().shift(-13))
        full_indret21d = pd.DataFrame(indret_nanfilled_df.rolling('21d').sum().shift(-20))
        full_indret30d = pd.DataFrame(indret_nanfilled_df.rolling('30d').sum().shift(-29))
        full_indret60d = pd.DataFrame(indret_nanfilled_df.rolling('60d').sum().shift(-59))
        full_indret90d = pd.DataFrame(indret_nanfilled_df.rolling('90d').sum().shift(-89))
        return_df = pd.DataFrame({\
            'ret_fwd2d_log':df[[]].merge(full_ret2d, how='left', left_index=True, right_index=True).values[:,0], \
            'ret_fwd7d_log':df[[]].merge(full_ret7d, how='left', left_index=True, right_index=True).values[:,0], \
            'ret_fwd14d_log':df[[]].merge(full_ret14d, how='left', left_index=True, right_index=True).values[:,0], \
            'ret_fwd21d_log':df[[]].merge(full_ret21d, how='left', left_index=True, right_index=True).values[:,0], \
            'ret_fwd30d_log':df[[]].merge(full_ret30d, how='left', left_index=True, right_index=True).values[:,0], \
            'ret_fwd60d_log':df[[]].merge(full_ret60d, how='left', left_index=True, right_index=True).values[:,0], \
            'ret_fwd90d_log':df[[]].merge(full_ret90d, how='left', left_index=True, right_index=True).values[:,0], \
            'indret_fwd2d_log':df[[]].merge(full_indret2d, how='left', left_index=True, right_index=True).values[:,0], \
            'indret_fwd7d_log':df[[]].merge(full_indret7d, how='left', left_index=True, right_index=True).values[:,0], \
            'indret_fwd14d_log':df[[]].merge(full_indret14d, how='left', left_index=True, right_index=True).values[:,0], \
            'indret_fwd21d_log':df[[]].merge(full_indret21d, how='left', left_index=True, right_index=True).values[:,0], \
            'indret_fwd30d_log':df[[]].merge(full_indret30d, how='left', left_index=True, right_index=True).values[:,0], \
            'indret_fwd60d_log':df[[]].merge(full_indret60d, how='left', left_index=True, right_index=True).values[:,0], \
            'indret_fwd90d_log':df[[]].merge(full_indret90d, how='left', left_index=True, right_index=True).values[:,0], \
            'ret_wrt_ind_fwd2d_log':df[[]].merge(full_ret2d, how='left', left_index=True, right_index=True).values[:,0], \
            'ret_wrt_ind_fwd7d_log':df[[]].merge(full_ret7d, how='left', left_index=True, right_index=True).values[:,0], \
            'ret_wrt_ind_fwd14d_log':df[[]].merge(full_ret14d, how='left', left_index=True, right_index=True).values[:,0], \
            'ret_wrt_ind_fwd21d_log':df[[]].merge(full_ret21d, how='left', left_index=True, right_index=True).values[:,0], \
            'ret_wrt_ind_fwd30d_log':df[[]].merge(full_ret30d, how='left', left_index=True, right_index=True).values[:,0], \
            'ret_wrt_ind_fwd60d_log':df[[]].merge(full_ret60d, how='left', left_index=True, right_index=True).values[:,0], \
            'ret_wrt_ind_fwd90d_log':df[[]].merge(full_ret90d, how='left', left_index=True, right_index=True).values[:,0], \
            })
        return_df.index = initial_index
        return return_df
    
    feature_data = stock_groups.apply(fwdret_based_func).reset_index(drop=True)
    feature_detail = {\
            'ret_fwd2d_log':{'description':'log-return for 2 days forward.', \
                'causality':False}, \
            'ret_fwd7d_log':{'description':'log-return for 7 days forward.', \
                'causality':False}, \
            'ret_fwd14d_log':{'description':'log-return for 14 days forward.', \
                'causality':False}, \
            'ret_fwd21d_log':{'description':'log-return for 21 days forward.', \
                'causality':False}, \
            'ret_fwd30d_log':{'description':'log-return for 30 days forward.', \
                'causality':True}, \
            'ret_fwd60d_log':{'description':'log-return for 60 days forward.', \
                'causality':False}, \
            'ret_fwd90d_log':{'description':'log-return for 90 days forward.', \
                'causality':False}, \
            'indret_fwd2d_log':{'description':'index log-return for 2 days forward.', \
                'causality':False}, \
            'indret_fwd7d_log':{'description':'index log-return for 7 days forward.', \
                'causality':False}, \
            'indret_fwd14d_log':{'description':'index log-return for 14 days forward.', \
                'causality':False}, \
            'indret_fwd21d_log':{'description':'index log-return for 21 days forward.', \
                'causality':False}, \
            'indret_fwd30d_log':{'description':'index log-return for 30 days forward.', \
                'causality':False}, \
            'indret_fwd60d_log':{'description':'index log-return for 60 days forward.', \
                'causality':False}, \
            'indret_fwd90d_log':{'description':'index log-return for 90 days forward.', \
                'causality':False}, \
            'ret_wrt_ind_fwd2d_log':{'description':'index log-return with respect to index for 2 days forward.', \
                'causality':False}, \
            'ret_wrt_ind_fwd7d_log':{'description':'index log-return with respect to index for 7 days forward.', \
                'causality':False}, \
            'ret_wrt_ind_fwd14d_log':{'description':'index log-return with respect to index for 14 days forward.', \
                'causality':False}, \
            'ret_wrt_ind_fwd21d_log':{'description':'index log-return with respect to index for 21 days forward.', \
                'causality':False}, \
            'ret_wrt_ind_fwd30d_log':{'description':'index log-return with respect to index for 30 days forward.', \
                'causality':False}, \
            'ret_wrt_ind_fwd60d_log':{'description':'index log-return with respect to index for 60 days forward.', \
                'causality':False}, \
            'ret_wrt_ind_fwd90d_log':{'description':'index log-return with respect to index for 90 days forward.', \
                'causality':False}, \
            }

    return feature_data, feature_detail

def bwd_prc_based_feature(merged_data_df):
    print('backward price based features')
    feature_data = pd.DataFrame([])
    feature_detail = dict()
    
    # grouping
    stock_groups = merged_data_df.groupby('stock_name')

    # map function
    def bwdret_based_func(stock_group):
        initial_index = stock_group.index
        df = stock_group
        df.index = [pd.Timestamp(item) for item in df['date']]

        full_ret1d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('1d').sum()
        full_ret2d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('2d').sum()
        full_ret3d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('3d').sum()
        full_ret7d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('7d').sum()
        full_ret14d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('14d').sum()
        full_ret21d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('21d').sum()
        full_ret30d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('30d').sum()
        full_ret60d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('60d').sum()
        full_ret90d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('90d').sum()
        full_ret120d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('120d').sum()
        full_ret300d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('300d').sum()

        full_indret1d = (np.log((df['total_stock_index']).pct_change() + 1)).rolling('1d').sum()
        full_indret2d = (np.log((df['total_stock_index']).pct_change() + 1)).rolling('2d').sum()
        full_indret3d = (np.log((df['total_stock_index']).pct_change() + 1)).rolling('3d').sum()
        full_indret7d = (np.log((df['total_stock_index']).pct_change() + 1)).rolling('7d').sum()
        full_indret14d = (np.log((df['total_stock_index']).pct_change() + 1)).rolling('14d').sum()
        full_indret21d = (np.log((df['total_stock_index']).pct_change() + 1)).rolling('21d').sum()
        full_indret30d = (np.log((df['total_stock_index']).pct_change() + 1)).rolling('30d').sum()
        full_indret60d = (np.log((df['total_stock_index']).pct_change() + 1)).rolling('60d').sum()
        full_indret90d = (np.log((df['total_stock_index']).pct_change() + 1)).rolling('90d').sum()
        full_indret120d = (np.log((df['total_stock_index']).pct_change() + 1)).rolling('120d').sum()
        full_indret300d = (np.log((df['total_stock_index']).pct_change() + 1)).rolling('300d').sum()
        return_df = pd.DataFrame({\
            'totalindex':df['total_stock_index'].values, \
            'indret_log':full_indret1d.values, \
            'indret_2d_log':full_indret2d.values, \
            'indret_3d_log':full_indret3d.values, \
            'indret_7d_log':full_indret7d.values, \
            'indret_14d_log':full_indret14d.values, \
            'indret_21d_log':full_indret21d.values, \
            'indret_30d_log':full_indret30d.values, \
            'indret_60d_log':full_indret60d.values, \
            'indret_90d_log':full_indret90d.values, \
            'indret_120d_log':full_indret120d.values, \
            'indret_300d_log':full_indret300d.values, \
            'ret_wrt_ind_log':full_ret1d.values - full_indret1d.values, \
            'ret_wrt_ind_2d_log':full_ret2d.values - full_indret2d.values, \
            'ret_wrt_ind_3d_log':full_ret3d.values - full_indret3d.values, \
            'ret_wrt_ind_7d_log':full_ret7d.values - full_indret7d.values, \
            'ret_wrt_ind_14d_log':full_ret14d.values - full_indret14d.values, \
            'ret_wrt_ind_21d_log':full_ret21d.values - full_indret21d.values, \
            'ret_wrt_ind_30d_log':full_ret30d.values - full_indret30d.values, \
            'ret_wrt_ind_60d_log':full_ret60d.values - full_indret60d.values, \
            'ret_wrt_ind_90d_log':full_ret90d.values - full_indret90d.values, \
            'ret_wrt_ind_120d_log':full_ret120d.values - full_indret120d.values, \
            'ret_wrt_ind_300d_log':full_ret300d.values - full_indret300d.values, \
            })
        return_df.index = initial_index
        return return_df
    
    feature_data = stock_groups.apply(bwdret_based_func).reset_index(drop=True)
    feature_detail = {\
            'indret_log':{'description':'index log-return.'}, \
            'indret_2d_log':{'description':'index log-return for 2 days.'}, \
            'indret_3d_log':{'description':'index log-return for 3 days.'}, \
            'indret_7d_log':{'description':'index log-return for 7 days.'}, \
            'indret_14d_log':{'description':'index log-return for 14 days.'}, \
            'indret_21d_log':{'description':'index log-return for 21 days.'}, \
            'indret_30d_log':{'description':'index log-return for 30 days.'}, \
            'indret_60d_log':{'description':'index log-return for 60 days.'}, \
            'indret_90d_log':{'description':'index log-return for 90 days.'}, \
            'indret_120d_log':{'description':'index log-return for 120 days.'}, \
            'indret_300d_log':{'description':'index log-return for 300 days.'}, \
            'ret_wrt_ind_log':{'description':'index log-return with respect to index.'}, \
            'ret_wrt_ind_2d_log':{'description':'index log-return with respect to index for 2 days.'}, \
            'ret_wrt_ind_3d_log':{'description':'index log-return with respect to index for 3 days.'}, \
            'ret_wrt_ind_7d_log':{'description':'index log-return with respect to index for 7 days.'}, \
            'ret_wrt_ind_14d_log':{'description':'index log-return with respect to index for 14 days.'}, \
            'ret_wrt_ind_21d_log':{'description':'index log-return with respect to index for 21 days.'}, \
            'ret_wrt_ind_30d_log':{'description':'index log-return with respect to index for 30 days.'}, \
            'ret_wrt_ind_60d_log':{'description':'index log-return with respect to index for 60 days.'}, \
            'ret_wrt_ind_90d_log':{'description':'index log-return with respect to index for 90 days.'}, \
            'ret_wrt_ind_120d_log':{'description':'index log-return with respect to index for 60 days.'}, \
            'ret_wrt_ind_300d_log':{'description':'index log-return with respect to index for 90 days.'}, \
            }

    return feature_data, feature_detail

