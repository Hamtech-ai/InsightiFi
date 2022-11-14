import datetime as datetime
from feature_extraction.indicator_features import *
from feature_extraction.daily_features import *  
import matplotlib.pyplot as plt
from model.model import *
import numpy as np
import pandas as pd

import pytse_client as tse
from pytse_client.download import download_financial_indexes
from sklearn.model_selection import train_test_split
from zigzag import peak_valley_pivots

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

    print(f'Shape of price features: {priceFeatures.shape}')

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

    print(f'Shape of indicator features: {indicatorFeatures.shape}')

    features = pd.concat(
        [
            priceFeatures,
            indicatorFeatures
        ],
        axis = 1
    )
    print(f'shape of features after concatenation: {features.shape}')

    features.dropna(
        axis = 'columns', 
        thresh = len(features) - 450,
        inplace = True
    )
    print(f'shape of features after drop non-essential columns: {features.shape}')
    print(f'names of the columns that dropped: {(indicatorFeatures.columns.union(priceFeatures.columns)).difference(features.columns).to_list()}')

    features.dropna(
        axis = 'index', 
        how = 'any',
        inplace = True
    )
    print(f'shape of features after drop non-essential rows: {features.shape}')


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
    # signals.replace(
    #     to_replace = -1, 
    #     value = 0,  
    #     inplace = True
    # )
    pivots.columns = ['pivots']
    signals['date'] = foladHist['date']
    pivots['date'] = foladHist['date']

    print(f'Number of each class {np.unique(signals["label"], return_counts = True)}')

    initFeatures = features.merge(
        signals,
        how = 'left',
        on = 'date'
    )

    print(f'Number of initial features: {initFeatures.shape}')

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
    
    def convertLabel(df, colName):
        name = str(colName + '01')
        df.loc[initFeatures[colName] == -1, name] = 0
        df[name] = df[name].fillna(1)
        return df

    initFeatures['priChange'] = initFeatures['adjClose'] - initFeatures['yesterday']
    labelPred = np.concatenate((trainPred, testPred), axis = 0)
    initFeatures['labelPred'] = labelPred
    labelProb = np.concatenate((trainProb[:, 1], testProb[:, 1]), axis = 0)
    initFeatures['labelProb'] = labelProb

    initFeatures = convertLabel(initFeatures, 'label')
    initFeatures = convertLabel(initFeatures, 'labelPred')

    output = initFeatures.merge(
        pivots,
        how = 'left',
        on = 'date'
    )

    output.to_csv(
        'outputFeatures.csv', 
        index = False
    )

    if output['labelPred'].iloc[-1] == 1:
        print(f'Buy Postition with {(output["labelProb"].iloc[-1]):0.0%} probability') 
    else:
        print(f'Sell Position with {(1 - output["labelProb"].iloc[-1]):0.0%} probability')