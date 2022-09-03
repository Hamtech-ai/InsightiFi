import numpy as np
import pandas as pd
import ta
#import copy

# Trend Indicator: Simple Moving Average (SMA)
def SMA(df):

    df['SMA10d'] = ta.trend.SMAIndicator(df['adjClose'], window = 10).sma_indicator()
    df['SMA20d'] = ta.trend.SMAIndicator(df['adjClose'], window = 20).sma_indicator()
    df['last_SMA20d'] = ta.trend.SMAIndicator(df['adjClose'], window = 20).sma_indicator().shift(1)
    df['SMA30d'] = ta.trend.SMAIndicator(df['adjClose'], window = 30).sma_indicator()
    df['SMA50d'] = ta.trend.SMAIndicator(df['adjClose'], window = 50).sma_indicator()
    df['last_SMA50d'] = ta.trend.SMAIndicator(df['adjClose'], window = 50).sma_indicator().shift(1)
    df['SMA80d'] = ta.trend.SMAIndicator(df['adjClose'], window = 80).sma_indicator()

    # buy signal
    df['RSI_buy'] = np.nan
    df.loc[(df['last_SMA20d'] < df['last_SMA50d']) & (df['SMA20d'] > df['SMA50d']), 'SMA_buy'] = 1
    # sell signal
    df.loc[(df['last_SMA20d'] > df['last_SMA50d']) & (df['SMA20d'] < df['SMA50d']), 'SMA_buy'] = 0

    df['SMA_position'] = df['SMA_buy'].fillna(method = 'ffill')

    SMAdf =  pd.DataFrame(
        {
        'SMA10d': df['SMA10d'],
        'SMA20d': df['SMA20d'],
        'last_SMA20d': df['last_SMA20d'],
        'SMA30d': df['SMA30d'],
        'SMA50d': df['SMA50d'],
        'last_SMA50d': df['last_SMA50d'],
        'SMA80d': df['SMA80d'],
        'SMA_buy': df['SMA_buy'],
        'SMA_position': df['SMA_position']
        }
    )

    return SMAdf


# Momentum Indicator: Relative Strength Index (RSI)
def RSI(df, window = 15):

    overBought = 70
    overSold = 30

    df['RSI'] = ta.momentum.RSIIndicator(df['adjClose'], window = window).rsi()
    df['last_RSI'] = df['RSI'].shift(1)

    # buy signal
    df['RSI_buy'] = np.nan
    df.loc[(df['last_RSI'] < overSold) & (df['RSI'] > overSold), 'RSI_buy'] = 1
    # sell signal
    df.loc[(df['last_RSI'] > overBought) & (df['RSI'] < overBought), 'RSI_buy'] = 0 

    df['RSI_position'] = df['RSI_buy'].fillna(method = 'ffill')


    RSIdf =  pd.DataFrame(
        {
        'RSI': df['RSI'],
        'RSI_buy': df['RSI_buy'],
        'RSI_position': df['RSI_position']
        }
    )

    return RSIdf
    