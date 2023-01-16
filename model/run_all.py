import datetime as datetime
import warnings

import numpy as np
import pandas as pd
import pytse_client as tse
from pytse_client.download import download_financial_indexes
from sklearn.model_selection import train_test_split
from zigzag import peak_valley_pivots

warnings.filterwarnings('ignore')
from feature_extraction.daily_features import *
from feature_extraction.indicator_features import *
from model import *

#########################

def runModel():

    tse.download(
        symbols = ['فولاد'],
        adjust = True, 
        write_to_csv = True, 
        include_jdate = True
    )
    folad = tse.Ticker(
        symbol = 'فولاد',  
        adjust = True
    )
    foladHist = folad.history
    foladClient = folad.client_types

    foladHist = foladHist[foladHist.columns[:-1].insert(0, foladHist.columns[-1]).to_list()] # change location of jdate

    ## Getting client data ##
    #########################

    foladClient = foladClient.iloc[::-1]
    foladClient.reset_index(
        drop = True,
        inplace = True
    )
    foladClient['date'] = foladClient['date'].apply(
        lambda x: datetime.datetime.strptime(x, '%Y%m%d')
    )
    foladClient.iloc[:, 1:] = foladClient.iloc[:, 1:].astype('float')

    ## Getting TEDPIX data ##
    #########################

    marketIndex = download_financial_indexes(
        symbols = 'شاخص كل', 
        write_to_csv = True
    )
    marketIndex = marketIndex['شاخص كل']
    marketIndex.rename(
        columns = {'value': 'TEDPIX'}, 
        inplace = True
    )
    foladHist = foladHist[foladHist.columns].merge(
        marketIndex, 
        how = 'left', 
        on = 'date'
    )

    ## Extracting features ##
    #########################

    calCols = calender_features(foladHist)
    candelCols = candlestick_feature(foladHist)
    prpCols = prp_based(foladHist)
    retCols = ret_based(foladHist)
    shiftCols = shift_data(foladHist)
    wghtCols = weight_feature(foladHist)

    priceFeatures = pd.concat(
        [
            foladHist,
            calCols,
            candelCols,
            prpCols,
            retCols,
            shiftCols,
            wghtCols
        ], axis = 1
    )

    priceFeatures = priceFeatures.merge(
        indv_nonindv_features(foladClient),
        how = 'left',
        on = 'date'
    )

    bbCols = BB(priceFeatures)
    emaCols = EMA(foladHist)
    macdCols = MACD(foladHist)
    smaCols = SMA(foladHist)
    stochasticCols = STOCHASTIC(foladHist)
    rsiCols = RSI(foladHist)

    indicatorFeatures = pd.concat(
        [
            bbCols,
            emaCols,
            macdCols,
            smaCols,
            stochasticCols,
            rsiCols,
        ], axis = 1
    )

    features = pd.concat(
        [
            priceFeatures,
            indicatorFeatures
        ],
        axis = 1
    )

    features.dropna(
        axis = 'columns', 
        thresh = len(features) - 450,
        inplace = True
    )

    features.dropna(
        axis = 'index', 
        how = 'any',
        inplace = True
    )


    ## Labeling data with zigzag ##
    ###############################

    pivots = pd.DataFrame(
        peak_valley_pivots(
            foladHist['adjClose'], 
            0.075, 
            -0.075
        ) * -1,
        columns = ['label']
    )
    signals = pivots.replace(
        to_replace = 0, 
        value = np.nan
    )
    signals.fillna(
        method = 'ffill', 
        inplace = True
    )

    pivots = pd.DataFrame(
        peak_valley_pivots(
            foladHist['adjClose'], 
            0.075, 
            -0.075
        ) * -1,
        columns = ['label']
    )
    signals = pivots.replace(
        to_replace = 0, 
        value = np.nan
    )
    signals.fillna(
        method = 'ffill', 
        inplace = True
    )
    pivots.columns = ['pivots']
    signals['date'] = foladHist['date']
    pivots['date'] = foladHist['date']

    initFeatures = features.merge(
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
    output.to_csv('../api/output_for_API.csv', index = False)
    
    return output

#########################

if __name__ == "__main__":
    output = runModel()