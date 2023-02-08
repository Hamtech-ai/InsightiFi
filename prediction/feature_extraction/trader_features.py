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

############################################
## Individual and Non-Individual Features ##
############################################
def indv_nonindv_features(ticker_client_types):

    df = ticker_client_types.copy()

    def daily_indv_nonindv_calc(df):
        daily_indv_nonindv_calc_df = pd.DataFrame(
            {   
                'date': df['date'],
                'enter_indv_money': df['individual_buy_value'] - df['individual_sell_value'],
                'indv_buy_per_capita': df['individual_buy_value'] / df['individual_buy_count'],
                'indv_sell_per_capita': df['individual_sell_value'] / df['individual_sell_count']
                
            }
        )
        return daily_indv_nonindv_calc_df
    
    daily_indv_nonindv_calc_df = daily_indv_nonindv_calc(df)
    daily_indv_nonindv_calc_df['indv_power'] = daily_indv_nonindv_calc_df['indv_buy_per_capita'] / daily_indv_nonindv_calc_df['indv_sell_per_capita']
    temp_daily_indv_nonindv_calc_df = pd.concat([daily_indv_nonindv_calc_df, calender_features(daily_indv_nonindv_calc_df)], axis = 1)
    
    def indv_nonindv_calc(df):
        indv_nonindv_calc_df = pd.DataFrame(
            {
                'enter_indv_money_1w': df['enter_indv_money'].rolling(5).sum(),
                'indv_buy_per_capita_daily_to_1w': df['indv_buy_per_capita'] / df['indv_buy_per_capita'].rolling(5).mean(),
                'indv_sell_per_capita_daily_to_1w': df['indv_sell_per_capita'] / df['indv_sell_per_capita'].rolling(5).mean(),
                'indv_power_daily_to_1w': df['indv_power'] / df['indv_power'].rolling(5).mean(),

                'enter_indv_money_2w': df['enter_indv_money'].rolling(10).sum(),
                'indv_buy_per_capita_daily_to_2w': df['indv_buy_per_capita'] / df['indv_buy_per_capita'].rolling(10).mean(),
                'indv_sell_per_capita_daily_to_2w': df['indv_sell_per_capita'] / df['indv_sell_per_capita'].rolling(10).mean(),
                'indv_power_daily_to_2w': df['indv_power'] / df['indv_power'].rolling(10).mean(),

                'enter_indv_money_3w': df['enter_indv_money'].rolling(15).sum(),
                'indv_buy_per_capita_daily_to_3w': df['indv_buy_per_capita'] / df['indv_buy_per_capita'].rolling(15).mean(),
                'indv_sell_per_capita_daily_to_3w': df['indv_sell_per_capita'] / df['indv_sell_per_capita'].rolling(15).mean(),
                'indv_power_daily_to_3w': df['indv_power'] / df['indv_power'].rolling(15).mean(),

                'enter_indv_money_1m': df.groupby(['year', 'month'])['enter_indv_money'].transform('sum'),
                'indv_buy_per_capita_daily_to_1m': df['indv_buy_per_capita'] / df.groupby(['year', 'month'])['indv_buy_per_capita'].transform('mean'),
                'indv_sell_per_capita_daily_to_1m': df['indv_sell_per_capita'] / df.groupby(['year', 'month'])['indv_sell_per_capita'].transform('mean'),
                'indv_power_daily_to_1m': df['indv_power'] / df.groupby(['year', 'month'])['indv_power'].transform('mean'),

                'enter_indv_money_3m': df.groupby(['year', 'quarter'])['enter_indv_money'].transform('sum'),
                'indv_buy_per_capita_daily_to_3m': df['indv_buy_per_capita'] / df.groupby(['year', 'quarter'])['indv_buy_per_capita'].transform('mean'),
                'indv_sell_per_capita_daily_to_3m': df['indv_sell_per_capita'] / df.groupby(['year', 'quarter'])['indv_sell_per_capita'].transform('mean'),
                'indv_power_daily_to_3m': df['indv_power'] / df.groupby(['year', 'quarter'])['indv_power'].transform('mean')
            }
        )
        return indv_nonindv_calc_df

    indv_nonindv_calc_df = indv_nonindv_calc(temp_daily_indv_nonindv_calc_df)
    return_df = pd.concat([daily_indv_nonindv_calc_df, indv_nonindv_calc_df], axis = 1)

    return return_df