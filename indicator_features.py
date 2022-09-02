import numpy as np
import pandas as pd
import ta
#import copy

def RSI(df, window = 15):

    if 'adjClose' not in df.columns:
        ValueError('your dataframe does not have a column named adjClose')
    
    overBought = 70
    overSold = 30

    df['RSI'] = ta.momentum.RSIIndicator(df['adjClose'], window = window).rsi()
    df['last_RSI'] = df['RSI'].shift(1)

    # buy signal
    df['RSI_buy'] = np.nan
    df.loc[(df['last_RSI'] < overBought) & (df['RSI'] > overBought), 'RSI_buy'] = 1
    df.loc[(df['last_RSI'] < overSold) & (df['RSI'] > overSold), 'RSI_buy'] = 1
    
    # sell signal
    df.loc[(df['last_RSI'] > overBought) & (df['RSI'] < overBought), 'RSI_buy'] = 0 
    df.loc[(df['last_RSI'] > overSold) & (df['RSI'] < overSold), 'RSI_buy'] = 0

    df['RSI_position'] = df['RSI_buy'].fillna(method = 'ffill')


    RSIdf =  pd.DataFrame(
        {
        'RSI': df['RSI'],
        'RSI_buy': df['RSI_buy'],
        'RSI_position': df['RSI_position']
        }
    )

    return RSIdf