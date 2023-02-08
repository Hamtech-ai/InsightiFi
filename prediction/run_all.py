import datetime as datetime
import warnings

import numpy as np
import pandas as pd
import pytse_client as tse
from pytse_client.download import download_financial_indexes
from sklearn.model_selection import train_test_split

from zigzag_indicator.zigzag import peak_valley_pivots

warnings.filterwarnings('ignore')
from feature_extraction.indicator_features import *
from feature_extraction.price_features import *
from feature_extraction.trader_features import *
from models.model import *

#########################

def runModel():

    tse.download(
        symbols = ['فولاد'],
        adjust = True, 
        write_to_csv = True, 
        include_jdate = True
    )
    ticker = tse.Ticker(
        symbol = 'فولاد',  
        adjust = True
    )
    ticker_history = ticker.history

    ticker_history = ticker_history[ticker_history.columns[:-1].insert(0, ticker_history.columns[-1]).to_list()] # change location of jdate

    ## Getting traders data ##
    #########################

    ticker_traders_types = ticker.client_types
    ticker_traders_types = ticker_traders_types.iloc[::-1]
    ticker_traders_types.reset_index(
        drop = True,
        inplace = True
    )
    ticker_traders_types['date'] = ticker_traders_types['date'].apply(
        lambda x: datetime.datetime.strptime(x, '%Y%m%d')
    )
    ticker_traders_types.iloc[:, 1:] = ticker_traders_types.iloc[:, 1:].astype('float')

    ## Getting TEDPIX data ##
    #########################

    market_index = download_financial_indexes(
        symbols = 'شاخص كل', 
        write_to_csv = True
    )
    market_index = market_index['شاخص كل']
    market_index.rename(
        columns = {'value': 'TEDPIX'}, 
        inplace = True
    )
    ticker_history = ticker_history.merge(
        market_index, 
        how = 'left', 
        on = 'date'
    )

    ## Extracting features ##
    #########################

    calenderFeatures = calender_features(ticker_history)
    candelFeatures = candlestick_feature(ticker_history)
    prpFeatures = prp_based(ticker_history)
    retFeatures = ret_based(ticker_history)
    shiftFeatures = shift_data(ticker_history)
    wghtFeatures = weight_feature(ticker_history)

    priceFeatures = pd.concat(
        [
            ticker_history,
            calenderFeatures,
            candelFeatures,
            prpFeatures,
            retFeatures,
            shiftFeatures,
            wghtFeatures
        ], axis = 1
    )

    traderFeatures = ticker_traders_types.merge(
        indv_nonindv_features(ticker_traders_types),
        how = 'left',
        on = 'date'
    )

    bbFeatures = BB(ticker_history)
    ichiFeatures = ICHIMOKU(ticker_history)
    emaFeatures = EMA(ticker_history)
    macdFeatures = MACD(ticker_history)
    smaFeatures = SMA(ticker_history)
    stochasticFeatures = STOCHASTIC(ticker_history)
    rsiFeatures = RSI(ticker_history)

    indicatorFeatures = pd.concat(
        [
            bbFeatures,
            ichiFeatures,
            emaFeatures,
            macdFeatures,
            smaFeatures,
            stochasticFeatures,
            rsiFeatures,
        ], axis = 1
    )


    features_extracted = pd.concat(
        [
            priceFeatures,
            indicatorFeatures
        ],
        axis = 1
    )
    features_extracted = features_extracted.merge(
        traderFeatures,
        how = 'right',
        on = 'date'
    )

    features_extracted.dropna(
        axis = 'columns', 
        thresh = len(features_extracted) - 385,
        inplace = True
    )

    features_extracted.dropna(
        axis = 'index', 
        how = 'any',
        inplace = True
    )



    ## Labeling data with zigzag ##
    ###############################

    pivots = pd.DataFrame(
        peak_valley_pivots(
            ticker_history['adjClose'].to_list(), 0.1
        ) ,
        columns = ['label']
    ) * -1

    signals = pivots.replace(
        to_replace = 0, 
        value = np.nan
    )
    signals.fillna(
        method = 'ffill', 
        inplace = True
    )
    signals.fillna(
        method = 'bfill', 
        inplace = True
    )

    pivots.columns = ['pivots']
    signals['date'] = ticker_history['date']
    pivots['date'] = ticker_history['date']

    initFeatures = features_extracted.merge(
        signals,
        how = 'left',
        on = 'date'
    )

    ## Modeling ##
    ##############

    X_train, X_test, y_train, y_test = train_test_split(
        initFeatures.iloc[:, 2:-1].values, 
        initFeatures.iloc[:, -1].values, 
        shuffle = False, 
        random_state = 0
    )
    trainPred, testPred, trainProb, testProb, featureImport, trainClassReport, testClassReport = RFClf(X_train, y_train, X_test, y_test)

    ## Getting output ##
    ####################
    
    output = pd.DataFrame()
    output['jdate'] = initFeatures['jdate']
    labelProb = np.concatenate((trainProb, testProb), axis = 0)[:,1]
    output['labelProb'] = labelProb
    try:
        output.to_csv('../api/inputs/output_for_API.csv', index = False) # if code run without docker
    except: 
        output.to_csv('output_for_API.csv', index = False) # if code run in docker
    print('alireza')
    return output

#########################

if __name__ == "__main__":
    output = runModel()