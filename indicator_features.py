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

# Volatility Indicator: Bollinger Bands (BB)
def BB(df):

    indicator_bb = ta.volatility.BollingerBands(df['adjClose'], window = 20, window_dev = 2)
    df['BB_bbh'] = indicator_bb.bollinger_hband()
    df['BB_bbl'] = indicator_bb.bollinger_lband()
    df['lag1d_BB_bbh'] = indicator_bb.bollinger_hband().shift(1)
    df['lag1d_BB_bbl'] = indicator_bb.bollinger_lband().shift(1)

    # buy signal
    df['BB_buy'] = np.nan
    df.loc[
        (df['lag1d_BB_bbl'] > df['lag1d_close']) & 
        (df['lag1d_open'] > df['lag1d_close'])&
        (df['BB_bbl'] > df['open']) & 
        (df['open'] < df['close']),
        'BB_buy' ] = 1
    # sell signal
    df.loc[
        (df['lag1d_BB_bbh'] < df['lag1d_close']) & 
        (df['lag1d_open'] < df['lag1d_close'])&
        (df['BB_bbh'] < df['open']) & 
        (df['open'] > df['close']),
        'BB_buy' ] = 0

    df['BB_position'] = df['BB_buy'].fillna(method = 'ffill')

    BBdf =  pd.DataFrame(
        {
            'BB_bbh': df['BB_bbh'],
            'BB_bbl': df['BB_bbl'],
            'lag1d_BB_bbh': df['lag1d_BB_bbh'],
            'lag1d_BB_bbl': df['lag1d_BB_bbl'],
            'BB_buy': df['BB_buy'],
            'BB_position': df['BB_position']
        }
    )

    return BBdf
