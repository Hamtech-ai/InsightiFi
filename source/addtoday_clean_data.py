import time

import pandas as pd
import numpy as np
import json
import requests
import re
import datetime as dt
today = dt.datetime.today().strftime('%Y-%m-%d')  

def convert_en_numbers(input_str):
    """
    Converts English numbers to Persian numbers
    :param input_str: String contains English numbers
    :return: New string with Persian numbers
    """
    mapping = {
        '0': '۰',
        '1': '۱',
        '2': '۲',
        '3': '۳',
        '4': '۴',
        '5': '۵',
        '6': '۶',
        '7': '۷',
        '8': '۸',
        '9': '۹',
        '.': '.',
    }
    return _multiple_replace(mapping, input_str)
def convert_en_characters(input_str):
    """
        Assumes that characters written with standard persian keyboard
        not windows arabic layout
    :param input_str: String contains English chars
    :return: New string with related characters on Persian standard keyboard
    """
    mapping = {
        'q': 'ض',
        'w': 'ص',
        'e': 'ث',
        'r': 'ق',
        't': 'ف',
        'y': 'غ',
        'u': 'ع',
        'i': 'ه',
        'o': 'خ',
        'p': 'ح',
        '[': 'ج',
        ']': 'چ',
        'a': 'ش',
        's': 'س',
        'd': 'ی',
        'f': 'ب',
        'g': 'ل',
        'h': 'ا',
        'j': 'ت',
        'k': 'ن',
        'l': 'م',
        ';': 'ک',
        "'": 'گ',
        'z': 'ظ',
        'x': 'ط',
        'c': 'ز',
        'v': 'ر',
        'b': 'ذ',
        'n': 'د',
        'm': 'پ',
        ',': 'و',
        '?': '؟',
    }
    return _multiple_replace(mapping, input_str)
def convert_ar_numbers(input_str):
    """
    Converts Arabic numbers to Persian numbers
    :param input_str: String contains Arabic numbers
    :return: New str and replaces arabic number with persian numbers
    """
    mapping = {
        '١': '۱',  # Arabic 1 is 0x661 and Persian one is 0x6f1
        '٢': '۲',  # More info https://goo.gl/SPiBtn
        '٣': '۳',
        '٤': '۴',
        '٥': '۵',
        '٦': '۶',
        '٧': '۷',
        '٨': '۸',
        '٩': '۹',
        '٠': '۰',
    }
    return _multiple_replace(mapping, input_str)
def convert_fa_numbers(input_str):
    """
    This function convert Persian numbers to English numbers.
    Keyword arguments:
    input_str -- It should be string
    Returns: English numbers
    """
    mapping = {
        '۰': '0',
        '۱': '1',
        '۲': '2',
        '۳': '3',
        '۴': '4',
        '۵': '5',
        '۶': '6',
        '۷': '7',
        '۸': '8',
        '۹': '9',
        '.': '.',
    }
    return _multiple_replace(mapping, input_str)
def convert_ar_characters(input_str):
    """
    Converts Arabic chars to related Persian unicode char
    :param input_str: String contains Arabic chars
    :return: New str with converted arabic chars
    """
    mapping = {
        'ك': 'ک',
        'دِ': 'د',
        'بِ': 'ب',
        'زِ': 'ز',
        'ذِ': 'ذ',
        'شِ': 'ش',
        'سِ': 'س',
        'ى': 'ی',
        'ي': 'ی'
    }
    return _multiple_replace(mapping, input_str)
def _multiple_replace(mapping, text):
    """
    Internal function for replace all mapping keys for a input string
    :param mapping: replacing mapping keys
    :param text: user input string
    :return: New string with converted mapping keys to values
    """
    pattern = "|".join(map(re.escape, mapping.keys()))
    return re.sub(pattern, lambda m: mapping[m.group()], str(text))

def Market_with_askbid():
    count = 0 
    while count<5:
        url = 'http://www.tsetmc.com/tsev2/data/MarketWatchPlus.aspx?h=0&r=0'
        data = requests.get(url, timeout=12)
        content = data.content.decode('utf-8')
        parts = content.split('@')
        if data.status_code != 200 or len(content.split('@')[2])<400:
            count+=1
            time.sleep(1)
        if count ==5:
            raise Exception('ohoh')
        if data.status_code == 200 and len(content.split('@')[2]) > 400:
            break
    parts = content.split('@')
    inst_price = parts[2].split(';')
    market_me = {}
    # Add the Trade and other stuff to dataframe--------
    for item in inst_price:
        item=item.split(',')
        market_me[item[0]]= dict(id=item[0],ISIN=item[1],symbol=item[2],
                              name=item[3],first_price=float(item[5]),close_price=float(item[6]),
                              last_trade=float(item[7]),count=item[8],volume=float(item[9]),
                              value=float(item[10]),min_traded_price=float(item[11]),
                              max_treaded_price=float(item[12]),yesterday_price=int(item[13]),
                              table_id=item[17],group_id=item[18],max_allowed_price=float(item[19]),
                              min_allowed_price=float(item[20]),last_ret = (float(item[7]) - float(item[13]))/float(item[13]),
                                 ret = (float(item[6]) - float(item[13]))/float(item[13]),
                                number_of_shares=float(item[21]), Market_cap=int(item[21]) *int(item[6]))
    # Add the Ask-Bid price Vol tu dataframe --------
    for item in parts[3].split(';'):
        try:
            item=item.split(',')
            if item[1] == '1':
                market_me[item[0]]['ask_price'.format(item[1])]=  float(item[4])
                market_me[item[0]]['ask_vol'.format(item[1])]=  float(item[6])
                market_me[item[0]]['bid_price'.format(item[1])]=  float(item[5])
                market_me[item[0]]['bid_vol'.format(item[1])]=  float(item[7])
        except:
            pass
    df = pd.DataFrame(market_me).T
    df = df[df['ISIN'].map(lambda x:not ( x.startswith('IRT')))]
    df = df[df['ISIN'].map(lambda x: not( x.startswith('IRB')))]
    df.symbol = df.symbol.map(lambda x: convert_ar_characters(x) )
    others = ['آ س پ' , 'جم پيلن', 'كي بي سي' , 'فن آوا' , 'انرژي3' , 'دتهران', 'های وب']
    df = df[df['symbol'].map(lambda x: True if (x in (others)) else x.isalpha())]
    df = df.set_index('symbol')
    return df

def indiv(all_data):
    count = 0
    while count <= 10:
        url = 'http://www.tsetmc.com/tsev2/data/ClientTypeAll.aspx'
        data = requests.get(url, timeout=10)
        content=data.content.decode('utf-8').split(";")
        if data.status_code != 200:
            count += 1
            time.sleep(0.5)
        elif data.status_code == 200:
            break
        elif count == 10:
            raise Exception('Loop Client!')
        else:
            time.sleep(1)
            pass
    # all_data = Market_with_askbid()
    all_data = all_data[all_data['ISIN'].map(lambda x:not ( x.startswith('IRT')))]
    all_data = all_data[all_data['ISIN'].map(lambda x:not ( x.startswith('IRB')))]
    others = ['آ س پ' , 'جم پيلن', 'كي بي سي' , 'فن آوا' , 'انرژي3' , 'دتهران', 'های وب']
    clienttype=[]
    for item in content:
        try:
            item=item.split(',')
            symbol = all_data[ all_data['id'] ==  item[0] ].index[0]
            clienttype.append(dict( id=item[0],Name = symbol,
                                       Individual_buy_count=int(item[1]),
                                          NonIndividual_buy_count=int(item[2]),
                                          Individual_buy_volume=int(item[3]),
                                          NonIndividual_buy_volume=int( item[4]) ,
                                          Individual_sell_count=int(item[5]),
                                          NonIndividual_sell_count=int(item[6]),
                                          Individual_sell_volume=int(item[7]),
                                          NonIndividual_sell_volume=int(item[8]),
                                         Value = float(all_data['value'][all_data.index == symbol].iloc[0]) ))
        except:
            continue
    clients = pd.DataFrame(clienttype)
    clients.Name = clients.Name.map(lambda x: convert_ar_characters(x) )
    clients['VAL_hoghooghi_SELL'] = clients['Value'] * clients['NonIndividual_sell_volume'].astype(float) /\
    (clients['Individual_sell_volume'].astype(float) +clients['NonIndividual_sell_volume'].astype(float))
    clients['VAL_hoghooghi_BUY'] = clients['Value'] * clients['NonIndividual_buy_volume'].astype(float) /\
    (clients['Individual_buy_volume'].astype(float) +clients['NonIndividual_buy_volume'].astype(float))
    clients['VAL_haghighi_BUY'] = clients['Value'] * clients['Individual_buy_volume'].astype(float) /\
    (clients['Individual_buy_volume'].astype(float) +clients['NonIndividual_buy_volume'].astype(float))
    clients['VAL_haghighi_SELL'] = clients['Value'] * clients['Individual_sell_volume'].astype(float) /\
    (clients['Individual_sell_volume'].astype(float) +clients['NonIndividual_sell_volume'].astype(float))
    clients['percapita_buy'] = clients['VAL_haghighi_BUY'] / clients['Individual_buy_count']
    clients['percapita_sell'] = clients['VAL_haghighi_SELL'] / clients['Individual_sell_count']
    clients['power'] = clients['percapita_buy'] / clients['percapita_sell']
    clients['VAL_net_haghigh'] = clients['VAL_haghighi_BUY'] - clients['VAL_haghighi_SELL']
    return clients     

df = Market_with_askbid()
df = indiv(df)

df = df.rename(columns={'Name':'stock_name' ,\
                        'VAL_hoghooghi_SELL':'NonIndividual_sell_value',\
                        'VAL_haghighi_BUY':'Individual_buy_value',\
                        'VAL_haghighi_SELL':'Individual_sell_value',\
                        'VAL_hoghooghi_BUY':'NonIndividual_buy_value'})
df = df[['stock_name','Individual_buy_count', 'NonIndividual_buy_count',\
        'Individual_sell_count', 'NonIndividual_sell_count',\
        'Individual_buy_volume', 'NonIndividual_buy_volume',\
        'Individual_sell_volume', 'NonIndividual_sell_volume',\
        'Individual_buy_value', 'NonIndividual_buy_value',\
        'Individual_sell_value', 'NonIndividual_sell_value']]

clean_df = pd.read_csv('data/clean_data/cleaned_daily_all.csv')

today_df = clean_df[clean_df['date']==today]

today_df = today_df.drop(['Individual_buy_count', 'NonIndividual_buy_count',\
        'Individual_sell_count', 'NonIndividual_sell_count',\
        'Individual_buy_volume', 'NonIndividual_buy_volume',\
        'Individual_sell_volume', 'NonIndividual_sell_volume',\
        'Individual_buy_value', 'NonIndividual_buy_value',\
        'Individual_sell_value', 'NonIndividual_sell_value'],axis=1)

new_df = today_df.merge(df, on='stock_name', how='left')

new_df = new_df.drop(['Unnamed: 0'],axis=1).reset_index(drop=True)
clean_df = clean_df.drop(['Unnamed: 0'],axis=1).reset_index(drop=True)
clean_df =  clean_df[clean_df['date']<today]
clean_df

final_df = pd.concat([clean_df,new_df],axis=0)

final_df.to_csv('data/clean_data/cleaned_daily_all.csv')
