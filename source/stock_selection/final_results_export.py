import pandas as pd
import numpy as np
import os as os
import datetime as dt
import json as json

def result_export(horizon, signal):

    ## saving date and stock name for results
    today = dt.datetime.today().strftime('%Y%m%d')  
    result_folder = 'data'+'/Predictions' + '/Ensemble_' + signal + '_Signal' + today +'.csv'
    saving_folder = 'data'+'/Predictions' + '/FinalScores' + signal + '_Signal' + today +'.xlsx'
    df = pd.read_csv(result_folder)
    # today_date = dt.datetime.today().strftime('%Y-%m-%d') 
    today_date = df.date.max()
    df = df[df['date']==today_date]
    df = df.drop(['Label', 'Unnamed: 0'],axis=1)
    df['Score'] = df['Score'].round(2)
    df = df.sort_values(by='Score', ascending = False)

    def color_mapping(v):

        if signal == 'Buy':

            if type(v) is not str:
                if 0.6<v<=1: 
                    color = '#32CD32' #lime green
                elif 0.5<v<=0.6: 
                    color = '#9ACD32' #yellow green
                elif 0.4<v<=0.5: 
                    color = '#FFFF00' #yellow 
                elif 0<=v<=0.4: 
                    color = '#F08080' #	light coral
                else:
                    color = 'white'

            elif type(v) is str:
                color = '#B0C4DE' #light steel blue
            else:
                color ='white'

        if signal == 'Sell':

            if type(v) is not str:
                if 0.6<v<=1: 
                    color = '#DC143C' #crimson
                elif 0.5<v<=0.6: 
                    color = '#FF6347' #tomato 
                elif 0.4<v<=0.5: 
                    color = '#FF7F50' #coral
                elif 0<=v<=0.4: 
                    color = '#FFA07A' #light salmon
                else:
                    color = 'white'

            elif type(v) is str:
                color = '#B0C4DE' #light steel blue
            else:
                color ='white'


        return 'background-color: %s' % color

    styler = (df.style.applymap(color_mapping))

    styler.to_excel(saving_folder)
    return 