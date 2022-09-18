#Daily Features

#Import
import pandas as pd
import numpy as np
import datetime as datetime
import pytse_client as tse
from pytse_client import download_client_types_records


#Price data
tickers = tse.download(symbols="فولاد", adjust=True, write_to_csv=True)
df = pd.DataFrame.from_dict(tickers["فولاد"])

#Individual data
if __name__ == '__main__':
    records_dict = download_client_types_records("فولاد")
df1 = pd.DataFrame.from_dict(records_dict["فولاد"])
#Rearrange from past to present with reindex
df1 = df1.iloc[::-1]
df1.index = np.arange(len(df1))

#Concat and align two recieved dataset in the initiate date
df.drop(df.index[0:348], inplace=True)
df_feature = pd.concat([df, df1], axis=1)



def candel_based_feature(df):
    print('weekly candles features')

    def candle_based(df):
        df.index = [pd.Timestamp(item) for item in df['date']]
        return_df = pd.DataFrame({\
            'date': df['date'],\
            'max_price7d': df['high'].rolling(5).max(),\
            'min_price7d': df['low'].rolling(5).min(),\
            'first_price7d': df['open'].rolling(5).agg(lambda rows: rows[0]),\
            'last_price7d': df['close'].rolling(5).agg(lambda rows: rows[-1]),\
            'yesterday_price7d': df['yesterday'].rolling(5).agg(lambda rows: rows[0]),\
            'max_price14d': df['high'].rolling(10).max(),\
            'min_price14d': df['low'].rolling(10).min(),\
            'first_price14d': df['open'].rolling(10).agg(lambda rows: rows[0]),\
            'last_price14d': df['close'].rolling(10).agg(lambda rows: rows[-1]),\
            'yesterday_price14d': df['yesterday'].rolling(10).agg(lambda rows: rows[0]),\
            'max_price21d': df['high'].rolling(15).max(),\
            'min_price21d': df['low'].rolling(15).min(),\
            'first_price21d': df['open'].rolling(15).agg(lambda rows: rows[0]),\
            'last_price21d': df['close'].rolling(15).agg(lambda rows: rows[-1]),\
            'yesterday_price21d': df['yesterday'].rolling(15).agg(lambda rows: rows[0]),\
            'max_price30d': df['high'].rolling(23).max(),\
            'min_price30d': df['low'].rolling(23).min(),\
            'first_price30d': df['open'].rolling(23).agg(lambda rows: rows[0]),\
            'last_price30d': df['close'].rolling(23).agg(lambda rows: rows[-1]),\
            'yesterday_price30d': df['yesterday'].rolling(23).agg(lambda rows: rows[0]),\
            'max_price60d': df['high'].rolling(44).max(),\
            'min_price60d': df['low'].rolling(44).min(),\
            'first_price60d': df['open'].rolling(44).agg(lambda rows: rows[0]),\
            'last_price60d': df['close'].rolling(44).agg(lambda rows: rows[-1]),\
            'yesterday_price60d': df['yesterday'].rolling(44).agg(lambda rows: rows[0]),\
            'max_price90d': df['high'].rolling(70).max(),\
            'min_price90d': df['low'].rolling(70).min(),\
            'first_price90d': df['open'].rolling(70).agg(lambda rows: rows[0]),\
            'last_price90d': df['close'].rolling(70).agg(lambda rows: rows[-1]),\
            'yesterday_price90d': df['yesterday'].rolling(70).agg(lambda rows: rows[0]),\
            })
        return return_df
    df_candle_based = candle_based(df)


    def shadow_based(df_candle_based):
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
        return return_df
    df_shadow_based = candle_based(df)

    df_feature = pd.concat([df_shadow_based, df_candle_based], axis=1)
    return df_feature
    
    
