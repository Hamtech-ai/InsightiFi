import pandas as pd
import numpy as np
import copy as copy
import datetime
import random

SELECTED_COLUMNS = [ 'level','indv_buy_pcap', 'indv_sell_pcap','nonindv_buy_pcap',\
                     'nonindv_sell_pcap','value','date', 'stock_name',\
                     'ret_fwd30d_log',\
                     'buy_queue_locked','sell_queue_locked',\
                     'ret_wrt_ind_log', 'ret_wrt_ind_2d_log','ret_wrt_ind_3d_log',\
                     'ret_wrt_ind_7d_log','ret_wrt_ind_14d_log','ret_wrt_ind_21d_log',\
                     'ret_wrt_ind_30d_log','ret_wrt_ind_60d_log',
                     'ret_wrt_ind_90d_log',\
                     'indv_sell_pcap_prpd7d30',\
                     'indv_sell_pcap_prpd14d60',\
                     'nonindv_buy_pcap_prpd7d30','nonindv_buy_pcap_prpd14d60',\
                     'nonindv_sell_pcap_prpd7d30',\
                     'nonindv_sell_pcap_prpd14d60',\
                     'ret1d_log','ret3d_log',\
                     'ret7d_log','ret14d_log','ret30d_log','ret60d_log',\
                     'ret90d_log',\
                     'prp_high30d','prp_high60d',\
                     'prp_high90d',\
                     'prp_low30d','prp_low60d',\
                     'prp_low90d',\
                     'prp_value3d30d','prp_value5d60d',\
                     'indv_net_count_prp7d30',\
                     'indv_relnet_value_d14',\
                     'indv_relnet_value_d30','indv_relnet_value_d60',\
                     'indv_relnet_value_d90',\
                     'nonindv_net_count_prp7d30',\
                     'nonindv_relnet_value_d14',\
                     'nonindv_relnet_value_d30','nonindv_relnet_value_d60',\
                     'nonindv_relnet_value_d90',\
                     'shadow_up7d', 'shadow_low7d', 'body7d','distance_from_sma10d',\
                    ]

def level_weight_features_edited(stock_daily_feature):

    df = stock_daily_feature[stock_daily_feature['level']==2]
    df = df.reset_index(drop=True)  
    df = df[SELECTED_COLUMNS]
    generated_df = pd.DataFrame()
    C=0
    while C != 60:
        
        stocks = random.choices(list(df['stock_name']),k=30)
        df_sel = df.loc[df['stock_name'].isin(stocks)]
        
        df_sel = df_sel.reset_index(drop=True)
        stock_groups = df_sel.groupby('stock_name')

        def value_weight_features(stock_group):
            initial_index = stock_group.index
            df = stock_group
            df.index = [pd.Timestamp(item) for item in df['date']]
            return_df = pd.DataFrame({\
                'value20d':df['value'].rolling('20d').sum()})
            return_df.index = initial_index
            return return_df
        feature_df = stock_groups.apply(value_weight_features).reset_index(drop=True)
        cols_to_go = ['date','level']
        feature_df = pd.concat([df_sel[cols_to_go],feature_df],axis=1).reset_index(drop=True)
        
        date_groups = feature_df.groupby('date')
        def value_weight(date_group):
            initial_index = date_group.index
            df = date_group
            return_df = pd.DataFrame({\
                            'value_weight' : df['value20d']/df['value20d'].sum()})
            return_df.index = initial_index
            return return_df
        value_weight_df = date_groups.apply(value_weight).reset_index(drop=True)
        
        feature_df = pd.concat([feature_df,value_weight_df],axis=1).reset_index(drop=True)
        
        stock_groups = df_sel.groupby('stock_name')
        def power_features(stock_group):
            initial_index = stock_group.index
            df = stock_group
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
                    /(df['nonindv_buy_pcap'] - df['nonindv_sell_pcap']).rolling('25d').sum()\
                                    })
            return_df.index = initial_index
            return return_df
        feature_data = stock_groups.apply(power_features).reset_index(drop=True)
        df_sel = df_sel.drop(['date','level'],axis=1)
        df_sel = pd.concat([df_sel,feature_data,feature_df],axis=1)


        exclude_cols = ['indv_buy_pcap', 'indv_sell_pcap', 'nonindv_buy_pcap',
                    'nonindv_sell_pcap', 'value', 'date', 'stock_name','level','value_weight']
        df_int = df_sel.drop(exclude_cols,axis=1)
        df_weight = pd.DataFrame()
        for i in df_int.columns:
            df_weight['lvl_'+i] = df_int[i]*df_sel['value_weight']
            
        col_sel = ['date','level']
        df_weight_date = pd.concat([df_sel[col_sel],df_weight],axis=1)
        
        final_df = pd.DataFrame()
        cols = df_weight.columns
        
        temp = df_weight_date.groupby(['date', 'level']).sum(min_count=1)
        for i in temp.columns:
            final_df['w'+i] = temp[i]
            
        generated_df = pd.concat([final_df,generated_df],axis=0)
        
        C = C+1
    generated_df = generated_df.reset_index()

    return generated_df
                