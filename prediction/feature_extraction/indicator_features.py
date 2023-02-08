import copy

import numpy as np
import pandas as pd
import ta

################################################
## Volatility Indicator: Bollinger Bands (BB) ##
################################################
def BB(ticker):
    
    df = ticker.copy()

    indicator_bb = ta.volatility.BollingerBands(df['adjClose'], window = 20, window_dev = 2)
    df['BB_bbh'] = indicator_bb.bollinger_hband()
    df['BB_bbl'] = indicator_bb.bollinger_lband()

    # buy(1) or sell(0) signal?
    df['BB_buy'] = np.nan
    # buy(1) signal
    df.loc[
        (df['BB_bbl'].shift() > df['close'].shift()) & 
        (df['open'].shift() > df['close'].shift()) &
        (df['BB_bbl'] > df['open']) & 
        (df['open'] < df['close']),
    'BB_buy' ] = 1
    # sell(0) signal
    df.loc[
        (df['BB_bbh'].shift() < df['close'].shift()) & 
        (df['open'].shift() < df['close'].shift()) &
        (df['BB_bbh'] < df['open']) & 
        (df['open'] > df['close']),
    'BB_buy' ] = 0

    df['BB_position'] = df['BB_buy'].fillna(method = 'ffill')

    return_df =  pd.DataFrame(
        {
            'BB_bbh': df['BB_bbh'],
            'BB_bbl': df['BB_bbl'],
            'BB_buy': df['BB_buy'],
            'BB_position': df['BB_position']
        }
    )

    return return_df

#######################################################
## Trend Indicator: Exponential Moving Average (EMA) ##
#######################################################
def EMA(ticker):

    df = ticker.copy()

    df['EMA_5d'] = ta.trend.ema_indicator(df['adjClose'], 5)
    df['last_EMA_5d'] = ta.trend.ema_indicator(df['adjClose'], 5).shift(1)
    df['EMA_40d'] = ta.trend.ema_indicator(df['adjClose'], 40)
    df['last_EMA_40d'] = ta.trend.ema_indicator(df['adjClose'], 40).shift(1)

    # buy(1) or sell(0) signal?
    df['EMA_buy'] = np.nan
    # buy(1) signal
    df.loc[
        (df['last_EMA_5d'] < df['last_EMA_40d']) & 
        (df['EMA_5d'] > df['EMA_40d']), 
    'EMA_buy'] = 1
    # sell(0) signal
    df.loc[
        (df['last_EMA_5d'] > df['last_EMA_40d']) & 
        (df['EMA_5d'] < df['EMA_40d']), 
    'EMA_buy'] = 0

    df['EMA_position'] = df['EMA_buy'].fillna(method = 'ffill')

    return_df =  pd.DataFrame(
        {
            'EMA_5d': df['EMA_5d'],
            'EMA_5d_diff': df['adjClose'] - df['EMA_5d'],
            'last_EMA_5d': df['last_EMA_5d'],
            'EMA_40d': df['EMA_40d'],
            'EMA_40d_diff': df['adjClose'] - df['EMA_40d'],
            'last_EMA_40d': df['last_EMA_40d'],
            'EMA_buy': df['EMA_buy'],
            'EMA_position': df['EMA_position']
        }
    )

    return return_df

####################################################
## Trend Indicator: Ichimoku Kinkō Hyō (Ichimoku) ##
####################################################
def ICHIMOKU(ticker):

    df = ticker.copy()

    df['Tenkan_Sen'] = ta.trend.ichimoku_conversion_line(df['high'], df['low']) # AKA: Turning Line
    df['Kijun_Sen'] = ta.trend.ichimoku_base_line(df['high'], df['low']) # AKA: Standard Line
    df['Chikou_Span']  = df['adjClose'].shift(-26) # AKA: Lagging Line
    df['Senkou_Span_A'] = ta.trend.ichimoku_a(df['high'], df['low']) # AKA: 1st Leading Line
    df['Senkou_Span_B'] = ta.trend.ichimoku_b(df['high'], df['low']) # AKA: 2nd Leading Line

    # buy(1) or sell(0) signal? (intersection Tenkan Sen and Kijun Sen)
    df['Tenkan_Kijun_cross_buy'] = np.nan
    # buy(1) signal
    df.loc[
        (df['Tenkan_Sen'].shift() < df['Kijun_Sen'].shift()) & 
        (df['Tenkan_Sen'] > df['Kijun_Sen']), 
    'Tenkan_Kijun_cross_buy'] = 1
    # sell(0) signal
    df.loc[
        (df['Tenkan_Sen'].shift() > df['Kijun_Sen'].shift()) & 
        (df['Tenkan_Sen'] < df['Kijun_Sen']), 
    'Tenkan_Kijun_cross_buy'] = 0

    df['Tenkan_Kijun_cross_position'] = df['Tenkan_Kijun_cross_buy'].fillna(method = 'ffill')

    # buy(1) or sell(0) signal? (intersection Senkou Span A and Senkou Span A)
    df['Senkou_cross_buy'] = np.nan
    # buy(1) signal
    df.loc[
        (df['Senkou_Span_A'].shift() < df['Senkou_Span_B'].shift()) & 
        (df['Senkou_Span_A'] > df['Senkou_Span_B']), 
    'Senkou_cross_buy'] = 1
    # sell(0) signal
    df.loc[
        (df['Senkou_Span_A'].shift() > df['Senkou_Span_B'].shift()) & 
        (df['Senkou_Span_A'] < df['Senkou_Span_B']), 
    'Senkou_cross_buy'] = 0

    df['Senkou_cross_position'] = df['Senkou_cross_buy'].fillna(method = 'ffill')

    # buy(1) or sell(0) signal? (intersection Chikou Span and adjClose)
    df['Chikou_adjClose_cross_buy'] = np.nan
    # buy(1) signal
    df.loc[
        (df['Chikou_Span'].shift() < df['adjClose'].shift()) & 
        (df['Chikou_Span'] > df['adjClose']), 
    'Chikou_adjClose_cross_buy'] = 1
    # sell(0) signal
    df.loc[
        (df['Chikou_Span'].shift() > df['adjClose'].shift()) & 
        (df['Chikou_Span'] < df['adjClose']), 
    'Chikou_adjClose_cross_buy'] = 0

    df['Chikou_adjClose_cross_position'] = df['Chikou_adjClose_cross_buy'].fillna(method = 'ffill')


    return_df =  pd.DataFrame(
        {
            'Tenkan_Sen': df['Tenkan_Sen'],
            'Kijun_Sen': df['Kijun_Sen'],
            'Chikou_Span': df['Chikou_Span'],
            'Senkou_Span_A': df['Senkou_Span_A'],
            'Senkou_Span_B': df['Senkou_Span_B'],
            'Tenkan_Kijun_cross_position': df['Tenkan_Kijun_cross_position'],
            'Senkou_cross_position': df['Senkou_cross_position'],
            'Chikou_adjClose_cross_position': df['Chikou_adjClose_cross_position']
        }
    )

    return return_df

###################################################################
## Trend Indicator: Moving Average Convergence Divergence (MACD) ##
###################################################################
def MACD(ticker):

    df = ticker.copy()

    df['MACD'] = ta.trend.macd(df['adjClose'], window_slow =26,  window_fast = 12) # ta.trend.ema_indicator(df['adjClose'], 12) - ta.trend.ema_indicator(df['adjClose'], 26)
    df['MACD_diff'] = ta.trend.macd_diff(df['adjClose'], window_slow = 26,  window_fast = 12) # df['MACD'] - df['MACD_signal']
    df['MACD_signal'] = ta.trend.macd_signal(df['adjClose'], window_slow = 26,  window_fast = 12) # ta.trend.ema_indicator(df['MACD'], 9)

    # buy(1) or sell(0) signal?
    df['MACD_buy'] = np.nan
    # buy(1) signal
    df.loc[
        (df['MACD'] < 0) & 
        (df['MACD'] > df['MACD_signal']) &
        (df['MACD'].shift(1) < df['MACD_signal'].shift(1)), 
    'MACD_buy'] = 1
    # sell(0) signal
    df.loc[
        (df['MACD'] > 0) &
        (df['MACD'] < df['MACD_signal']) &
        (df['MACD'].shift(1) > df['MACD_signal'].shift(1)), 
    'MACD_buy'] = 0

    df['MACD_position'] = df['MACD_buy'].fillna(method = 'ffill')

    return_df =  pd.DataFrame(
        {
            'MACD': df['MACD'],
            'MACD_diff': df['MACD_diff'],
            'MACD_signal': df['MACD_signal'],
            'MACD_buy': df['MACD_buy'],
            'MACD_position': df['MACD_position']
        }
    )

    return return_df

#######################################################
## Momentum Indicator: Relative Strength Index (RSI) ##
#######################################################
def RSI(ticker, window = 15):

    df = ticker.copy()
    overBought = 70
    overSold = 30

    df['RSI'] = ta.momentum.RSIIndicator(df['adjClose'], window = window).rsi()
    df['last_RSI'] = df['RSI'].shift(1)

    # buy(1) or sell(0) signal?
    df['RSI_buy'] = np.nan
    # buy(1) signal
    df.loc[
        (df['last_RSI'] < overSold) &
        (df['RSI'] > overSold), 
    'RSI_buy'] = 1
    # sell(0) signal
    df.loc[
        (df['last_RSI'] > overBought) &
        (df['RSI'] < overBought), 
    'RSI_buy'] = 0 

    df['RSI_position'] = df['RSI_buy'].fillna(method = 'ffill')

    return_df =  pd.DataFrame(
        {
            'RSI': df['RSI'],
            'RSI_buy': df['RSI_buy'],
            'RSI_position': df['RSI_position']
        }
    )

    return return_df

##################################################
## Trend Indicator: Simple Moving Average (SMA) ##
##################################################
def SMA(ticker):

    df = ticker.copy()

    df['SMA10d'] = ta.trend.SMAIndicator(df['adjClose'], window = 10).sma_indicator()
    df['SMA20d'] = ta.trend.SMAIndicator(df['adjClose'], window = 20).sma_indicator()
    df['last_SMA20d'] = ta.trend.SMAIndicator(df['adjClose'], window = 20).sma_indicator().shift(1)
    df['SMA30d'] = ta.trend.SMAIndicator(df['adjClose'], window = 30).sma_indicator()
    df['SMA50d'] = ta.trend.SMAIndicator(df['adjClose'], window = 50).sma_indicator()
    df['last_SMA50d'] = ta.trend.SMAIndicator(df['adjClose'], window = 50).sma_indicator().shift(1)
    df['SMA80d'] = ta.trend.SMAIndicator(df['adjClose'], window = 80).sma_indicator()

    # buy(1) or sell(0) signal?
    df['RSI_buy'] = np.nan
    # buy(1) signal
    df.loc[
        (df['last_SMA20d'] < df['last_SMA50d']) & 
        (df['SMA20d'] > df['SMA50d']), 
    'SMA_buy'] = 1
    # sell(0) signal
    df.loc[
        (df['last_SMA20d'] > df['last_SMA50d']) & 
        (df['SMA20d'] < df['SMA50d']), 
    'SMA_buy'] = 0

    df['SMA_position'] = df['SMA_buy'].fillna(method = 'ffill')

    return_df =  pd.DataFrame(
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

    return return_df

####################################################
## Momentum Indicator: Stochastic Oscillator (SR) ##
####################################################
def STOCHASTIC(ticker):

    df = ticker.copy()
    overBought = 70
    overSold = 20

    df['STOCH_fast'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], 15, 5).stoch()
    df['STOCH_slow'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], 15, 5).stoch_signal()
    df['last_STOCH_fast'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], 15, 5).stoch().shift(1)
    df['last_STOCH_slow'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], 15, 5).stoch_signal().shift(1)

    # buy(1) or sell(0) signal?
    df['STOCH_buy'] = np.nan
    # buy(1) signal
    df.loc[
        (df['STOCH_fast'] > overSold) &
        (df['last_STOCH_fast'] < df['last_STOCH_slow']) &
        (df['STOCH_slow'] < df['STOCH_fast']), 
    'STOCH_buy'] = 1
    # sell(0) signal
    df.loc[
        (df['STOCH_fast'] > overBought) &
        (df['last_STOCH_fast'] > df['last_STOCH_slow']) &
        (df['STOCH_slow'] > df['STOCH_fast']), 
    'STOCH_buy'] = 0 

    df['STOCH_position'] = df['STOCH_buy'].fillna(method = 'ffill').shift()

    return_df =  pd.DataFrame(
        {
            'STOCH_fast': df['STOCH_fast'],
            'STOCH_slow': df['STOCH_slow'],
            'last_STOCH_fast': df['last_STOCH_fast'],
            'last_STOCH_slow': df['last_STOCH_slow'],
            'STOCH_buy': df['STOCH_buy'],
            'STOCH_position': df['STOCH_position']
        }
    )
    
    return return_df