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


# Trend Indicator: Exponential Moving Average (EMA)
def EMA(df):

    df['EMA_5d'] = ta.trend.ema_indicator(df['adjClose'], 5)
    df['last_EMA_5d'] = ta.trend.ema_indicator(df['adjClose'], 5).shift(1)
    df['EMA_40d'] = ta.trend.ema_indicator(df['adjClose'], 40)
    df['last_EMA_40d'] = ta.trend.ema_indicator(df['adjClose'], 40).shift(1)

    # buy signal
    df['EMA_buy'] = np.nan

    df.loc[(df['last_EMA_5d'] < df['last_EMA_40d']) & (df['EMA_5d'] > df['EMA_40d']), 'EMA_buy'] = 1
    # sell signal
    df.loc[(df['last_EMA_5d'] > df['last_EMA_40d']) & (df['EMA_5d'] < df['EMA_40d']), 'EMA_buy'] = 0

    df['EMA_position'] = df['EMA_buy'].fillna(method = 'ffill')

    EMAdf =  pd.DataFrame(
        {
            'EMA_5d': df['EMA_5d'],
            'last_EMA_5d': df['last_EMA_5d'],
            'EMA_40d': df['EMA_40d'],
            'last_EMA_40d': df['last_EMA_40d'],
            'EMA_buy': df['EMA_buy'],
            'EMA_position': df['EMA_position']
        }
    )

    return EMAdf


# Trend Indicator: Moving Average Convergence Divergence (MACD)
def MACD(df):

    df['MACD'] = ta.trend.macd(df['adjClose'], window_slow =26,  window_fast = 12) # ta.trend.ema_indicator(df['adjClose'], 12) - ta.trend.ema_indicator(df['adjClose'], 26)
    df['MACD_diff'] = ta.trend.macd_diff(df['adjClose'], window_slow = 26,  window_fast = 12) # df['MACD'] - df['MACD_signal']
    df['MACD_signal'] = ta.trend.macd_signal(df['adjClose'], window_slow = 26,  window_fast = 12) # ta.trend.ema_indicator(df['MACD'], 9)

    # buy signal
    df['MACD_buy'] = np.nan
    df.loc[(df['MACD'] < 0) & (df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) < df['MACD_signal'].shift(1)), 'MACD_buy'] = 1
    # sell signal
    df.loc[(df['MACD'] > 0) & (df['MACD'] < df['MACD_signal']) & (df['MACD'].shift(1) > df['MACD_signal'].shift(1)), 'MACD_buy'] = 0

    df['MACD_position'] = df['MACD_buy'].fillna(method = 'ffill')

    MACDdf =  pd.DataFrame(
    {'adjClose': df['adjClose'],
        'MACD': df['MACD'],
        'MACD_diff': df['MACD_diff'],
        'MACD_signal': df['MACD_signal'],
        'MACD_buy': df['MACD_buy'],
        'MACD_position': df['MACD_position']
    }
    )

    return MACDdf


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


# Momentum Indicator: Stochastic Oscillator (SR)
def STOCHASTIC(df):

    overBought = 80
    overSold = 20

    df['STOCH_osci'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], 15, 5).stoch()
    df['STOCH_signal'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], 15, 5).stoch_signal()
    df['last_STOCH_osci'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], 15, 5).stoch().shift(1)
    df['last_STOCH_signal'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], 15, 5).stoch_signal().shift(1)

    # buy signal
    df['STOCH_buy'] = np.nan
    df.loc[
        (df['STOCH_osci'] < overSold) &
        (df['last_STOCH_osci'] > overSold) &
        (df['STOCH_signal'] < df['STOCH_osci']), 
        'STOCH_buy'] = 1
    # sell signal
    df.loc[
        (df['STOCH_osci'] < overBought) &
        (df['last_STOCH_osci'] > overBought) &
        (df['STOCH_signal'] > df['STOCH_osci']), 
        'STOCH_buy'] = 0 

    df['STOCH_position'] = df['STOCH_buy'].fillna(method = 'ffill')

    STOCHASTICdf =  pd.DataFrame(
    {
        'STOCH_osci': df['STOCH_osci'],
        'STOCH_signal': df['STOCH_signal'],
        'last_STOCH_osci': df['last_STOCH_osci'],
        'last_STOCH_signal': df['last_STOCH_signal'],
        'STOCH_buy': df['STOCH_buy'],
        'STOCH_position': df['STOCH_position']
    }
    )
    
    return STOCHASTICdf 
       

# Volatility Indicator: Bollinger Bands (BB)
def BB(df):

    indicator_bb = ta.volatility.BollingerBands(df['adjClose'], window = 20, window_dev = 2)
    df['BB_bbh'] = indicator_bb.bollinger_hband()
    df['BB_bbl'] = indicator_bb.bollinger_lband()
    df['last_BB_bbh'] = indicator_bb.bollinger_hband().shift(1)
    df['last_BB_bbl'] = indicator_bb.bollinger_lband().shift(1)

    # buy signal
    df['BB_buy'] = np.nan
    df.loc[
        (df['last_BB_bbl'] > df['last_close']) & 
        (df['last_open'] > df['last_close'])&
        (df['BB_bbl'] > df['open']) & 
        (df['open'] < df['close']),
        'BB_buy' ] = 1
    # sell signal
    df.loc[
        (df['last_BB_bbh'] < df['last_close']) & 
        (df['last_open'] < df['last_close'])&
        (df['BB_bbh'] < df['open']) & 
        (df['open'] > df['close']),
        'BB_buy' ] = 0

    df['BB_position'] = df['BB_buy'].fillna(method = 'ffill')

    BBdf =  pd.DataFrame(
        {
            'BB_bbh': df['BB_bbh'],
            'BB_bbl': df['BB_bbl'],
            'last_BB_bbh': df['last_BB_bbh'],
            'last_BB_bbl': df['last_BB_bbl'],
            'BB_buy': df['BB_buy'],
            'BB_position': df['BB_position']
        }
    )

    return BBdf
