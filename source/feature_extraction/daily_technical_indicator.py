from configs.param_getter import technical_parameters

import numpy as np
import pandas as pd
import ta
import copy

def sma_based_features(stock_daily_data):
    print('sma based features')
    feature_data = pd.DataFrame([])
    feature_detail = dict()

    # grouping
    stock_groups = stock_daily_data.groupby('stock_name')

    # map function
    def sma_diff_tp(stock_group):
        initial_index = stock_group.index
        df = stock_group
        df['typical_price'] = (df['adj_close_price']+df['adj_max_price']+df['adj_min_price'])/3
        df['SMA10d'] = df['adj_close_price'].rolling(window=10).mean()
        df['SMA21d'] = df['adj_close_price'].rolling(window=21).mean()
        df['SMA30d'] = df['adj_close_price'].rolling(window=30).mean()
        df['SMA60d'] = df['adj_close_price'].rolling(window=60).mean()
        df['SMA90d'] = df['adj_close_price'].rolling(window=90).mean()
        return_df = pd.DataFrame({\
            'distance_from_sma10d': (df['SMA10d'] - df['typical_price'])/df['SMA10d'],\
            'distance_from_sma21d': (df['SMA21d'] - df['typical_price'])/df['SMA21d'],\
            'distance_from_sma30d': (df['SMA30d'] - df['typical_price'])/df['SMA30d'],\
            'distance_from_sma60d': (df['SMA60d'] - df['typical_price'])/df['SMA60d'],\
            'distance_from_sma90d': (df['SMA90d'] - df['typical_price'])/df['SMA90d'] })
        return return_df
    sma_data = stock_groups.apply(sma_diff_tp).reset_index(drop=True)

    sma_detail = {\
        'distance_from_sma10d': {'description': 'distance of 10-day sma of close price from daily typical price'},\
        'distance_from_sma21d': {'description': 'distance of 21-day sma of close price from daily typical price'},\
        'distance_from_sma30d': {'description': 'distance of 30-day sma of close price from daily typical price'},\
        'distance_from_sma60d': {'description': 'distance of 60-day sma of close price from daily typical price'},\
        'distance_from_sma90d': {'description': 'distance of 90-day sma of close price from daily typical price'}}

    return sma_data, sma_detail


def ichimoku_based_feature(stock_daily_data):
    print('ichimoku based features')
    feature_data = pd.DataFrame([])
    feature_detail = dict()

    # grouping
    stock_groups = stock_daily_data.groupby('stock_name')

    # map function
    def tkc_based_func(stock_group):
        initial_index = stock_group.index
        df = stock_group
        period9_high = df['adj_max_price'].rolling(window=9).max()
        period9_low  = df['adj_min_price'].rolling(window=9).min()
        df['tenkan_sen'] = (period9_high + period9_low)/2
        ###########
        period26_high = df['adj_max_price'].rolling(window=26).max()
        period26_low  = df['adj_min_price'].rolling(window=26).min()
        df['kijun_sen'] = (period26_high + period26_low)/2
        ###########
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen'])/2).shift(26)
        ###########
        period52_high = df['adj_max_price'].rolling(window=52).max()
        period52_low  = df['adj_min_price'].rolling(window=52).min()
        df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)
        df['chikou_span'] = df['adj_close_price'].shift(-26)
        
        df['above_cloud'] = 0
        df['above_cloud'] = np.logical_and(df['adj_min_price'] > df['senkou_span_a'], df['adj_min_price'] > df['senkou_span_b'])
        df['under_cloud'] = np.logical_and(df['adj_max_price'] < df['senkou_span_a'], df['adj_max_price'] < df['senkou_span_b'])
        df['A_above_B'] = np.where((df['senkou_span_a'] > df['senkou_span_b']), 1, -1)
        
        df['lines_ascending'] = np.logical_and(df['tenkan_sen'] > df['tenkan_sen'].shift(1),
                                        df['kijun_sen'] > df['kijun_sen'].shift(1))
        df['lines_descending'] = np.logical_and(df['tenkan_sen'] < df['tenkan_sen'].shift(1),
                                        df['kijun_sen'] < df['kijun_sen'].shift(1))

        df['chikou_span_above_price'] = np.logical_and(df['chikou_span'] > df['adj_max_price'], True)
        df['chikou_span_under_price'] = np.logical_and(df['chikou_span'] < df['adj_min_price'], True)

        df['chikou_span_buy'] = np.logical_and(np.logical_and(df['above_cloud'], df['lines_ascending']),
                                        df['chikou_span_above_price']).astype(int)
        df['chikou_span_sell'] = np.logical_and(np.logical_and(df['under_cloud'], df['lines_descending']),
                                        df['chikou_span_under_price']).astype(int)

        return_df = pd.DataFrame({\
            'base_line_difference':df['adj_min_price'] - df['kijun_sen'], \
            'chikou_span_sell':df['chikou_span_sell'], \
            'chikou_span_buy':df['chikou_span_buy'], \
            })
        return_df.index = initial_index
        return return_df
    
    feature_data = stock_groups.apply(tkc_based_func).reset_index(drop=True)
    feature_detail = {\
            'base_line_difference':{'description':'min price difference from base line'}, \
            'chikou_span_sell':{'description':'chikou span sell signal.', \
                'best_horizon':'more than 30 days'}, \
            'chikou_span_buy':{'description':'chikou span buy signal.', \
                'best_horizon':'more than 30 days'}, \
            } 

    return feature_data, feature_detail

def cci_based_feature(stock_daily_data):
    print('cci based features')
    feature_data = pd.DataFrame([])
    feature_detail = dict()

    # params 
    params = technical_parameters('CCI')
    n_cci = params['n_cci']
    up_th_cci = params['up_th_cci']
    dn_th_cci = params['dn_th_cci']

    # grouping
    stock_groups = stock_daily_data.groupby('stock_name')

    # map function
    def cci_based_func(stock_group):
        initial_index = stock_group.index
        df = copy.copy(stock_group)
        df['cci'] = ta.trend.cci(df['adj_max_price'], df['adj_min_price'], df['adj_close_price'], \
            n=n_cci)

        # signal: if condittion1, then 1. otherwise, 0. 
        df['cci_buy_signal'] = np.logical_and(df['cci'] >  up_th_cci, df['cci'].shift(1) <  up_th_cci).astype(int)

        # signal: if condittion1, then 1. otherwise, 0. 
        df['cci_sell_signal'] = np.logical_and(df['cci'] < dn_th_cci, df['cci'].shift(1) > dn_th_cci).astype(int)
        return_df = pd.DataFrame({\
            'cci':df['cci'], \
            'cci_buy_signal':df['cci_buy_signal'], \
            'cci_sell_signal':df['cci_sell_signal'], \
            })
        return_df.index = initial_index
        return return_df
    
    feature_data = stock_groups.apply(cci_based_func).reset_index(drop=True)
    feature_detail = {\
            'cci':{'description':'commodity channel index.'}, \
            'cci_buy_signal':{'description':'cci buy signal.'}, \
            'cci_sell_signal':{'description':'cci sell signal.'}, \
            } 

    return feature_data, feature_detail

def rvi_based_feature(stock_daily_data):
    print('rvi based features')
    feature_data = pd.DataFrame([])
    feature_detail = dict()

    # params 
    params = technical_parameters('RVI')
    n_rvi = params['n_rvi']
    n_std_rvi = params['n_std_rvi']
    th_rvi = params['th_rvi']

    # grouping
    stock_groups = stock_daily_data.groupby('stock_name')

    # map function
    def rvi_based_func(stock_group):
        initial_index = stock_group.index
        df = copy.copy(stock_group)

        df['std_h'] = df['adj_max_price'].rolling(n_std_rvi).std()
        df['std_l'] = df['adj_min_price'].rolling(n_std_rvi).std()
        
        # high
        up_series = df['std_h']*(df['adj_max_price'] >= df['adj_max_price'].shift(1)).astype(int)
        up_series.iloc[0:min(n_rvi, len(up_series))] = np.zeros([min(n_rvi, len(up_series))])

        dn_series = df['std_h']*(df['adj_max_price'] < df['adj_max_price'].shift(1)).astype(int)
        dn_series.iloc[0:min(n_rvi, len(dn_series))] = np.zeros([min(n_rvi, len(dn_series))])
        
        df['upavg_h'] = np.convolve(up_series, \
            (1/n_rvi)*np.power(1-1/n_rvi, np.arange(len(up_series))), mode='same')
        df['dnavg_h'] = np.convolve(dn_series, \
            (1/n_rvi)*np.power(1-1/n_rvi, np.arange(len(up_series))), mode='same')
        
        # low
        up_series = df['std_l']*(df['adj_min_price'] >= df['adj_min_price'].shift(1)).astype(int)
        up_series.iloc[0:min(n_rvi, len(up_series))] = np.zeros([min(n_rvi, len(up_series))])

        dn_series = df['std_l']*(df['adj_min_price'] < df['adj_min_price'].shift(1)).astype(int)
        dn_series.iloc[0:min(n_rvi, len(dn_series))] = np.zeros([min(n_rvi, len(dn_series))])
        
        df['upavg_l'] = np.convolve(up_series, \
            (1/n_rvi)*np.power(1-1/n_rvi, np.arange(len(up_series))), mode='same')
        df['dnavg_l'] = np.convolve(dn_series, \
            (1/n_rvi)*np.power(1-1/n_rvi, np.arange(len(up_series))), mode='same')
        
        #
        df['rvi_orig_high'] = (100 * df['upavg_h'] / (df['upavg_h'] + df['dnavg_h'])).replace([np.NAN], 0)
        df['rvi_orig_low']  = (100 * df['upavg_l'] / (df['upavg_l'] + df['dnavg_l'])).replace([np.NAN], 0)
        df['rvi'] = (df['rvi_orig_high'] + df['rvi_orig_low']) / 2

        # buy signal: if condittion1, then 1. otherwise, 0. 
        df['rvi_buy_signal'] = np.logical_and(df['rvi'] >  (100 - th_rvi), df['rvi'].shift(1) <  (100 - th_rvi)).astype(int)

        # sell signal: if condittion1, then 1. otherwise, 0. 
        df['rvi_sell_signal'] = np.logical_and(df['rvi'] < th_rvi, df['rvi'].shift(1) > th_rvi).astype(int)

        return_df = pd.DataFrame({\
            'rvi':df['rvi'], \
            'rvi_buy_signal':df['rvi_buy_signal'], \
            'rvi_sell_signal':df['rvi_sell_signal'], \
            })
        return_df.index = initial_index
        return return_df
    
    feature_data = stock_groups.apply(rvi_based_func).reset_index(drop=True)
    feature_detail = {\
            'rvi':{'description':'relative volatility index.'}, \
            'rvi_buy_signal':{'description':'rvi buy signal.', \
                'best_horizon':'less than 15 days'}, \
            'rvi_sell_signal':{'description':'rvi sell signal.', \
                'best_horizon':'less than 15 days'}, \
            }

    return feature_data, feature_detail

def roc_based_feature(stock_daily_data):
    print('roc based features')
    feature_data = pd.DataFrame([])
    feature_detail = dict()

    # params 
    params = technical_parameters('ROC')
    n_roc = params['n_roc']
    th_roc = params['th_roc']

    # grouping
    stock_groups = stock_daily_data.groupby('stock_name')

    # map function
    def roc_based_func(stock_group):
        initial_index = stock_group.index
        df = copy.copy(stock_group)

        df['roc'] = ta.momentum.roc(df['adj_close_price'], n=n_roc)

        # buy signal: if condittion1, then 1. otherwise, 0. 
        df['roc_buy_signal'] = np.logical_and(df['roc'] > th_roc, df['roc'].shift(1) < th_roc).astype(int)

        # sell signal: if condittion1, then 1. otherwise, 0. 
        df['roc_sell_signal'] = np.logical_and(df['roc'] < - th_roc, df['roc'].shift(1) > - th_roc).astype(int)

        return_df = pd.DataFrame({\
            'roc':df['roc'], \
            'roc_buy_signal':df['roc_buy_signal'], \
            'roc_sell_signal':df['roc_sell_signal'], \
            })
        return_df.index = initial_index
        return return_df
    
    feature_data = stock_groups.apply(roc_based_func).reset_index(drop=True)
    feature_detail = {\
            'roc':{'description':'rate of change'}, \
            'roc_buy_signal':{'description':'roc buy signal.', \
                'best_horizon':'less than 10 days'}, \
            'roc_sell_signal':{'description':'roc sell signal.', \
                'best_horizon':'less than 10 days'}, \
            }
    return feature_data, feature_detail

def mfi_based_feature(stock_daily_data):
    print('mfi based features')
    feature_data = pd.DataFrame([])
    feature_detail = dict()

    # params 
    params = technical_parameters('MFI')
    n_mfi = params['n_mfi']
    th_mfi = params['th_mfi']

    # grouping
    stock_groups = stock_daily_data.groupby('stock_name')

    # map function
    def mfi_based_func(stock_group):
        initial_index = stock_group.index
        df = copy.copy(stock_group)

        df['mfi'] = ta.volume.money_flow_index(df['adj_max_price'], df['adj_min_price'], df['adj_close_price'], df['adj_volume'], n=n_mfi)

        # buy signal: if condittion1, then 1.otherwise, 0. 
        df['mfi_buy_signal'] = np.logical_and(df['mfi'] >  (100 - th_mfi), df['mfi'].shift(1) <  (100 - th_mfi)).astype(int)

        # sell signal: if condittion1, then 1. otherwise, 0. 
        df['mfi_sell_signal'] = np.logical_and(df['mfi'] < th_mfi, df['mfi'].shift(1) > th_mfi).astype(int)

        return_df = pd.DataFrame({\
            'mfi':df['mfi'], \
            'mfi_buy_signal':df['mfi_buy_signal'], \
            'mfi_sel_signal':df['mfi_sell_signal'], \
            })
        return_df.index = initial_index
        return return_df
    
    feature_data = stock_groups.apply(mfi_based_func).reset_index(drop=True)
    feature_detail = {\
            'mfi':{'description':'money flow index'}, \
            'mfi_buy_signal':{'description':'mfi buy signal.', \
                'best_horizon':'less than 15 days'}, \
            'mfi_sell_signal':{'description':'mfi sell signal.', \
                'best_horizon':'less than 15 days'}, \
            }

    return feature_data, feature_detail

def cci_light_based_feature(stock_daily_data):
    print('cci light based features')
    feature_data = pd.DataFrame([])
    feature_detail = dict()

    # params 
    params = technical_parameters('CCI_light')
    n_cci_light = params['n_cci_light']
    up_th_cci_light = params['up_th_cci_light']
    below_saturate_cci_light = params['below_saturate_cci_light']

    # grouping
    stock_groups = stock_daily_data.groupby('stock_name')

    # map function
    def cci_light_based_func(stock_group):
        initial_index = stock_group.index
        df = copy.copy(stock_group)
        df['cci_light'] = ta.trend.cci(df['adj_max_price'], df['adj_min_price'], df['adj_close_price'], \
            n=n_cci_light)
        df['cci_light'] = np.where(df['cci_light'] > below_saturate_cci_light, df['cci_light'], below_saturate_cci_light) \
            - up_th_cci_light

        # signal: if condittion1, then 1. otherwise, 0. 
        df['cci_light_buy_signal'] = (df['cci_light'] > up_th_cci_light).astype(int)
        return_df = pd.DataFrame({\
            'cci_light':df['cci_light'], \
            'cci_light_buy_signal':df['cci_light_buy_signal'], \
            })
        return_df.index = initial_index
        return return_df
    
    feature_data = stock_groups.apply(cci_light_based_func).reset_index(drop=True)
    feature_detail = {\
            'cci_light':{'description':'commodity channel index with longer period.'}, \
            'cci_light_buy_signal':{'description':'cci-light buy signal.'}, \
            }

    return feature_data, feature_detail

def ema_light_based_feature(stock_daily_data):
    print('ema light based features')
    feature_data = pd.DataFrame([])
    feature_detail = dict()

    # params 
    params = technical_parameters('EMA_light')
    n_ema_light = params['n_ema_light']

    # grouping
    stock_groups = stock_daily_data.groupby('stock_name')

    # map function
    def ema_light_based_func(stock_group):
        initial_index = stock_group.index
        df = copy.copy(stock_group)

        df['ema_light'] = ta.trend.ema_indicator(df['adj_close_price'], n=n_ema_light)

        # signal: if condittion1, then 1. otherwise, 0. 
        df['ema_light_buy_signal'] = (df['adj_close_price'] > df['ema_light']).astype(int)
        
        return_df = pd.DataFrame({\
            'ema_light':df['ema_light'], \
            'ema_light_buy_signal':df['ema_light_buy_signal'], \
            })
        return_df.index = initial_index
        return return_df
    
    feature_data = stock_groups.apply(ema_light_based_func).reset_index(drop=True)
    feature_detail = {\
            'ema_light':{'description':'exponential moving average for longer period'}, \
            'ema_light_buy_signal':{'description':'ema-light buy signal.'}, \
            }

    return feature_data, feature_detail

def ema_based_feature(stock_daily_data):
    print('ema based features')
    feature_data = pd.DataFrame([])
    feature_detail = dict()

    # params 
    params = technical_parameters('EMA')
    n_short_ema = params['n_short_ema']
    n_long_ema = params['n_long_ema']

    # grouping
    stock_groups = stock_daily_data.groupby('stock_name')

    # map function
    def ema_based_func(stock_group):
        initial_index = stock_group.index
        df = copy.copy(stock_group)

        df['short_ema'] = ta.trend.ema_indicator(df['adj_close_price'], n=n_short_ema, fillna=False) 
        df['long_ema']  = ta.trend.ema_indicator(df['adj_close_price'], n=n_long_ema, fillna=False)    
        df['ema_buy']  = np.logical_and(df['long_ema'] < df['short_ema'],
                                        df['long_ema'].shift(1) > df['short_ema'].shift(1)).astype(int)
        df['ema_sell'] = np.logical_and(df['long_ema'] > df['short_ema'],
                                        df['long_ema'].shift(1) < df['short_ema'].shift(1)).astype(int) 
        
        return_df = pd.DataFrame({\
            'ema_buy':df['ema_buy'], \
            'ema_sell':df['ema_sell'], \
            })
        return_df.index = initial_index
        return return_df
    
    feature_data = stock_groups.apply(ema_based_func).reset_index(drop=True)
    feature_detail = {\
            'ema_buy':{'description':'ema sell signal', \
                'best_horizon':'less than 10 days'}, \
            'ema_sell':{'description':'ema buy signal', \
                'best_horizon':'less than 10 days'}, \
            }

    return feature_data, feature_detail

def aroon_based_feature(stock_daily_data):
    print('aroon based features')
    feature_data = pd.DataFrame([])
    feature_detail = dict()

    # params 
    params = technical_parameters('AROON')
    n_aroon = params['n_aroon']
    th_aroon = params['th_aroon']

    # grouping
    stock_groups = stock_daily_data.groupby('stock_name')

    # map function
    def aroon_based_func(stock_group):
        initial_index = stock_group.index
        df = copy.copy(stock_group)

        df['aroonup'] = ta.trend.aroon_up(df['adj_close_price'], n=n_aroon)
        df['aroondn'] = ta.trend.aroon_down(df['adj_close_price'], n=n_aroon)
        df['aroon']   = df['aroonup'] - df['aroondn']    
        df['aroon_buy'] = np.logical_and(df['aroon'] > th_aroon, df['aroon'].shift(1) < th_aroon).astype(int)
        df['aroon_sell'] = np.logical_and(df['aroon'] < th_aroon, df['aroon'].shift(1) > th_aroon).astype(int)     
        
        return_df = pd.DataFrame({\
            'aroon_buy':df['aroon_buy'], \
            'aroon_sell':df['aroon_sell'], \
            })
        return_df.index = initial_index
        return return_df
    
    feature_data = stock_groups.apply(aroon_based_func).reset_index(drop=True)
    feature_detail = {\
            'aroon_buy':{'description':'aroon sell signal', \
                'best_horizon':'less than 10 days'}, \
            'aroon_sell':{'description':'aroon buy signal', \
                'best_horizon':'less than 10 days'}, \
            }

    return feature_data, feature_detail

def rsi_based_feature(stock_daily_data):
    print('rsi based features')
    feature_data = pd.DataFrame([])
    feature_detail = dict()

    # params 
    params = technical_parameters('RSI')
    n_rsi = params['n_rsi']
    th_rsi = params['th_rsi']

    # grouping
    stock_groups = stock_daily_data.groupby('stock_name')

    # map function
    def rsi_based_func(stock_group):
        initial_index = stock_group.index
        df = copy.copy(stock_group)

        df['rsi'] = ta.momentum.rsi(df['adj_close_price'], n=n_rsi)
        df['rsi_buy']  = np.logical_and(df['rsi'] > (100-th_rsi), df['rsi'].shift(1) < (100-th_rsi)).astype(int)
        df['rsi_sell'] = np.logical_and(df['rsi'] < th_rsi, df['rsi'].shift(1) > th_rsi).astype(int)
        
        return_df = pd.DataFrame({\
            'rsi':df['rsi'], \
            'rsi_buy':df['rsi_buy'], \
            'rsi_sell':df['rsi_sell'], \
            })
        return_df.index = initial_index
        return return_df
    
    feature_data = stock_groups.apply(rsi_based_func).reset_index(drop=True)
    feature_detail = {\
            'rsi':{'description':'rsi signal'}, \
            'rsi_buy':{'description':'rsi sell signal', \
                'best_horizon':'good for many horizons'}, \
            'rsi_sell':{'description':'rsi buy signal', \
                'best_horizon':'good for many horizons'}, \
            }

    return feature_data, feature_detail

def uo_based_feature(stock_daily_data):
    print('uo based features')
    feature_data = pd.DataFrame([])
    feature_detail = dict()

    # params 
    params = technical_parameters('UO')
    th_uo = params['th_uo']
    s = params['s']
    m = params['m']
    len_uo = params['len_uo']
    ws = params['ws']
    wm = params['wm']
    wl = params['wl']

    # grouping
    stock_groups = stock_daily_data.groupby('stock_name')

    # map function
    def uo_based_func(stock_group):
        initial_index = stock_group.index
        df = copy.copy(stock_group)

        df['uo'] = ta.momentum.uo(df['adj_max_price'], df['adj_min_price'], df['adj_close_price'], \
            s=s, m=m, len=len_uo, ws=ws, wm=wm, wl=wl)
        df['uo_buy']  = np.logical_and(df['uo'] > (100-th_uo), df['uo'].shift(1) < (100-th_uo)).astype(int)
        df['uo_sell'] = np.logical_and(df['uo'] < th_uo, df['uo'].shift(1) > th_uo).astype(int)      
        
        return_df = pd.DataFrame({\
            'uo':df['uo'], \
            'uo_buy':df['uo_buy'], \
            'uo_sell':df['uo_sell'], \
            })
        return_df.index = initial_index
        return return_df
    
    feature_data = stock_groups.apply(uo_based_func).reset_index(drop=True)
    feature_detail = {\
            'uo':{'description':'uo signal'}, \
            'uo_buy':{'description':'uo sell signal', \
                'best_horizon':'good for many horizons'}, \
            'uo_sell':{'description':'uo buy signal', \
                'best_horizon':'good for many horizons'}, \
            }

    return feature_data, feature_detail

def stoch_based_feature(stock_daily_data):
    print('stoch based features')
    feature_data = pd.DataFrame([])
    feature_detail = dict()

    # params 
    params = technical_parameters('STOCH')
    n_stoch = params['n_stoch']
    n_d_stoch = params['n_d_stoch']
    th_stoch = params['th_stoch']

    # grouping
    stock_groups = stock_daily_data.groupby('stock_name')

    # map function
    def stoch_based_func(stock_group):
        initial_index = stock_group.index
        df = copy.copy(stock_group)

        df['stoch_osci'] = ta.momentum.stoch_signal(df['adj_max_price'], df['adj_min_price'], df['adj_close_price'], \
            n=n_stoch, d_n=n_d_stoch)
        df['stoch_D'] = df['stoch_osci'].rolling(3).mean()
        df['stoch_buy']  = np.logical_and(np.logical_and(df['stoch_D'] > (100-th_stoch),
            df['stoch_D'].shift(1) < (100-th_stoch)), df['stoch_D'] < df['stoch_osci']).astype(int)
        df['stoch_sell'] = np.logical_and(np.logical_and(df['stoch_D'] < th_stoch,
            df['stoch_D'].shift(1) > th_stoch), df['stoch_D'] > df['stoch_osci']).astype(int)
                                        
        return_df = pd.DataFrame({\
            'stoch_osci':df['stoch_osci'], \
            'stoch_buy':df['stoch_buy'], \
            'stoch_sell':df['stoch_sell'], \
            })
        return_df.index = initial_index
        return return_df
    
    feature_data = stock_groups.apply(stoch_based_func).reset_index(drop=True)
    feature_detail = {\
            'stoch_osci':{'description':'stoch oscilation signal'}, \
            'stoch_buy':{'description':'stoch sell signal', \
                'best_horizon':'less than 10 days'}, \
            'stoch_sell':{'description':'stoch buy signal', \
                'best_horizon':'less than 10 days'}, \
            }

    return feature_data, feature_detail

def obv_based_feature(stock_daily_data):
    print('obv based features')
    feature_data = pd.DataFrame([])
    feature_detail = dict()

    # params 
    params = technical_parameters('OBV')
    n_short_obv = params['n_short_obv']
    n_long_obv = params['n_long_obv']
    n_minmax_obv = params['n_minmax_obv']

    # grouping
    stock_groups = stock_daily_data.groupby('stock_name')

    # map function
    def obv_based_func(stock_group):
        initial_index = stock_group.index
        df = copy.copy(stock_group)

        df['obv'] = ta.volume.on_balance_volume(df['adj_close_price'], df['adj_volume'])
        df['obv_mean_short'] = df['obv'].rolling(n_short_obv).mean()
        df['obv_mean_long']  = df['obv'].rolling(n_long_obv).mean()
        df['close_max'] = df['adj_close_price'].rolling(n_minmax_obv).max()
        df['close_min'] = df['adj_close_price'].rolling(n_minmax_obv).min()
        df['obv_buy'] = np.logical_and(\
            np.logical_and(df['adj_close_price'] == df['close_min'], df['close_max'] != df['close_min']), \
            np.logical_and(df['obv'] > df['obv_mean_short'], df['obv_mean_short'] > df['obv_mean_long'])).astype(int)
        df['obv_sell'] = np.logical_and(\
            np.logical_and(df['adj_close_price'] == df['close_max'], df['close_max'] != df['close_min']),
            np.logical_and(df['obv'] < df['obv_mean_short'], df['obv_mean_short'] < df['obv_mean_long'])).astype(int)    
                                        
        return_df = pd.DataFrame({\
            'obv_buy':df['obv_buy'], \
            'obv_sell':df['obv_sell'], \
            })
        return_df.index = initial_index
        return return_df
    
    feature_data = stock_groups.apply(obv_based_func).reset_index(drop=True)
    feature_detail = {\
            'obv_buy':{'description':'obv sell signal', \
                'best_horizon':'more than 10 days'}, \
            'obv_sell':{'description':'obv buy signal', \
                'best_horizon':'more than 10 days'}, \
            }

    return feature_data, feature_detail

def macd_based_feature(stock_daily_data):
    print('macd based features')
    feature_data = pd.DataFrame([])
    feature_detail = dict()

    # params 
    params = technical_parameters('MACD')
    n_slow_macd = params['n_slow_macd']
    n_fast_macd = params['n_fast_macd']
    n_sign_macd = params['n_sign_macd']

    # grouping
    stock_groups = stock_daily_data.groupby('stock_name')

    # map function
    def macd_based_func(stock_group):
        initial_index = stock_group.index
        df = copy.copy(stock_group)

        df['macd'] = ta.trend.macd(df['adj_close_price'], n_slow=n_slow_macd, n_fast=n_fast_macd)
        df['macd_signalline'] = ta.trend.macd_signal(df['adj_close_price'], n_slow=n_slow_macd,
                                                    n_fast=n_fast_macd, n_sign=n_sign_macd)
        df['macd_buy']  = np.logical_and(df['macd'].shift(1) > df['macd_signalline'].shift(1),
                                np.logical_and(df['macd'] < df['macd_signalline'], df['macd'] > 0)).astype(int)
        df['macd_sell'] = np.logical_and(df['macd'].shift(1) < df['macd_signalline'].shift(1),
                                np.logical_and(df['macd'] > df['macd_signalline'], df['macd'] < 0)).astype(int)

        return_df = pd.DataFrame({\
            'macd':df['macd'], \
            'macd_signalline':df['macd_signalline'], \
            'macd_buy':df['macd_buy'], \
            'macd_sell':df['macd_sell'], \
            })
        return_df.index = initial_index
        return return_df
    
    feature_data = stock_groups.apply(macd_based_func).reset_index(drop=True)
    feature_detail = {\
            'macd':{'description':'macd signal'}, \
            'macd_signalline':{'description':'macd signal line'}, \
            'macd_buy':{'description':'macd sell signal', \
                'best_horizon':'more than 15 days'}, \
            'macd_sell':{'description':'macd buy signal', \
                'best_horizon':'more than 15 days'}, \
            }

    return feature_data, feature_detail