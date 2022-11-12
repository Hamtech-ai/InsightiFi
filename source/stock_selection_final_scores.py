import pandas as pd
import numpy as np
import os as os
import datetime as dt
import json as json
import psycopg2
import pandas.io.sql as sqlio
import re
import requests
from requests_toolbelt import MultipartEncoder
def ershan(isin):
    URLp = 'https://www.nahayatnegar.com/tv/{}'.format(isin)
    return '<a href="{}"> -{}</a>'.format(URLp, '(ارشن)')
def URL(id):
    if type(id) == str:
        URLp = 'http://www.tsetmc.com/Loader.aspx?ParTree=151311&i={}'.format(id)
    else: 
        id = str(id)
        URLp = 'http://www.tsetmc.com/Loader.aspx?ParTree=151311&i={}'.format(id)
    return '<a href="{}"> -{}</a>'.format(URLp, '(TSETMC)')
def _multiple_replace(mapping, text):
    """
    Internal function for replace all mapping keys for a input string
    :param mapping: replacing mapping keys
    :param text: user input string
    :return: New string with converted mapping keys to values
    """
    pattern = "|".join(map(re.escape, mapping.keys()))
    return re.sub(pattern, lambda m: mapping[m.group()], str(text))
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
def tahlicator(msg , chat_id = "-1001468757086"):
    headers = {'Content-type': 'application/json'}
    payload = {"bot-name" : "hermes" , 
              "chat-id" : "-1001468757086",
              "message" : msg,
              "parse-mode" : "html"}
    r = requests.post('http://178.62.251.62:8891/send-message', 
                      headers = headers,
                      data=json.dumps(payload))
    if r.status_code == 200:
        return True
def telegram_msg_kimia(msg , chat_id = "626243872"):
    headers = {'Content-type': 'application/json'}
    payload = {"bot-name" : "hermes" , 
              "chat-id" : chat_id,
              "message" : msg,
              "parse-mode" : "html"}
    r = requests.post('http://178.62.251.62:8891/send-message', 
                      headers = headers,
                      data=json.dumps(payload))
def telegram_msg_mehrdad(msg , chat_id = "-1001470501669"):
    headers = {'Content-type': 'application/json'}
    payload = {"bot-name" : "hermes" , 
              "chat-id" : "417161976",
              "message" : msg,
              "parse-mode" : "html"}
    r = requests.post('http://178.62.251.62:8891/send-message', 
                      headers = headers,
                      data=json.dumps(payload))
def Market_with_askbid():
    count = 0
    while count<5:
        url = 'http://www.tsetmc.com/tsev2/data/MarketWatchPlus.aspx?h=0&r=0'
        data = requests.get(url, timeout=12)
        content = data.content.decode('utf-8')
        parts = content.split('@')
        if data.status_code != 200 or len(content.split('@')[2]) < 500:
            count+=1
            time.sleep(1)
        if count ==5:
            raise Exception('ohoh')
        if data.status_code == 200 and len(content.split('@')[2]) > 500:
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
                              min_allowed_price=float(item[20]),
                              last_ret = (float(item[7]) - float(item[13]))/float(item[13]),
                                 ret = (float(item[6]) - float(item[13]))/float(item[13]),
                                number_of_shares=float(item[21]), Market_cap=int(item[21]) *int(item[6]))
    # Add the Ask-Bid price Vol to dataframe --------
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
    df = df[df['ISIN'].map(lambda x: not(x.startswith('IRT')))]
    df = df[df['ISIN'].map(lambda x: not(x.startswith('IRB')))]
    others = ['آ س پ' , 'جم پيلن', 'كي بي سي' , 'فن آوا' , 'انرژي3' , 'دتهران', 'هاي وب']
    df = df[df['symbol'].map(lambda x: True if (x in (others)) else x.isalpha())]
    df.symbol = df.symbol.map(lambda x: convert_ar_characters(x))
    df = df.set_index('symbol', drop=False)
    return df

def send_pdf_telegram (pdf, caption, chat_id):
    fields = {"bot-name": "hermes", "chat-id": chat_id, "pdf": (pdf, open(pdf, "rb")), 
             "parse-mode": "html", "caption": caption}
    url = 'http://159.223.11.86:8891/send-pdf'
    m = MultipartEncoder(fields, boundary = 'my_super_custom_header')
    r = requests.post(url, headers = {'Content-Type': m.content_type}, data = m.to_string())
today = dt.datetime.today().strftime('%Y%m%d')  
cwd = os.getcwd()
buy_location = cwd + '/data'+'/Predictions' + '/Ensemble_' + 'Buy' + '_Signal' + today +'.csv'
sell_location = cwd +  '/data'+'/Predictions' + '/Ensemble_' + 'Sell' + '_Signal' + today +'.csv'

df_buy = pd.read_csv(buy_location)
df_sell = pd.read_csv(sell_location)
today_date = df_buy.date.max()
df_buy = df_buy[df_buy['date']==today_date]
df_buy = df_buy.drop(['Label', 'Unnamed: 0'],axis=1)
df_buy['Score'] = df_buy['Score'].round(2)
df_buy = df_buy.sort_values(by='Score', ascending = False)
df_buy = df_buy.reset_index(drop = True)
df_buy.rename  (columns = {'Score':'buy_score'}, inplace = True)

df_sell = df_sell[df_sell['date']==today_date]
df_sell = df_sell.drop(['Label', 'Unnamed: 0'],axis=1)
df_sell['Score'] = df_sell['Score'].round(2)
df_sell = df_sell.sort_values(by='Score', ascending = False)
df_sell = df_sell.reset_index(drop = True)
df_sell.rename  (columns = {'Score':'sell_score'}, inplace = True)

df = df_buy.merge(df_sell, on = ["date", "stock_name"])

conn =  psycopg2.connect(host='37.152.186.128', port = 5432, database='aftermarket', \
                               user='postgres', password='admin123!@#')
cursor = conn.cursor()
cursor.executemany("insert into ml_scores(date, stock_name, buy_score, sell_score) values (%s, %s, %s, %s)" , df.values.tolist() )
conn.commit()
conn.close()

today = str(dt.datetime.now().date()).replace('-','')

#window = pd.read_excel('FinalScores_'+today+'.xlsx')


conn    =  psycopg2.connect(host='37.152.186.128', port = 5432, database='aftermarket', user='postgres', password='admin123!@#')
cursor  =  conn.cursor()
sql     = sql = "select date, stock_name, buy_score, sell_score, buy_score - sell_score as net_score \
from ml_scores where DATE(date) = DATE(NOW()) order by net_score DESC"

conn.commit()
window = sqlio.read_sql_query(sql, conn)
if len(window) != 0:
           
    window['stock_name'] = window['stock_name'].map(lambda x: x.replace(' ', '_'))
    good_list = []
    good_list = window[(window['net_score'] >= 0.55)].head(20)['stock_name'].to_list()
    mc = Market_with_askbid()
    mc.index = mc.index.map(lambda x: x.replace(' ', '_'))
    count = window[(window['net_score'] >= 0.55)].shape[0]
    if count >0 :
        selected = mc.loc[mc.index.isin(good_list)]
        selected = selected[selected['Market_cap']> 1e13]
        selected = selected.sort_values(by='Market_cap', ascending=False)



        if count >20:
            String= ' ⚙️💹🧠💻⚙️لیست بیست سهم برتر دارای موقعیت خرید برای افق ۳۰ روزه از دید خردجمعی مدلهای یادگیری عمیق: \n\n '
        else:
            String = ' ⚙️💹🧠💻⚙️لیست سهم های برتر دارای موقعیت خرید برای افق ۳۰ روزه از دید خردجمعی مدلهای یادگیری عمیق: \n\n '
        for item in good_list:
            try:
                String += '#' + item + URL(selected['id'][selected.index==item].iloc[0]) + '  ' + ershan(selected['ISIN'][selected.index==item].iloc[0]) +'\n'
            except:
                pass
        String += '\n              ➖➖➖ هرمس ➖➖➖  '
        
        telegram_msg_mehrdad(String)
        telegram_msg_kimia(String)
        tahlicator(String)  
    else:
        String = "برای امروز مدل هوش مصنوعی سهمی را پیشنهاد نمیدهد."
        String +='\n              ➖➖➖ هرمس ➖➖➖  '
        tahlicator(String)
        print("no signal for today")

     
        
        
    window[["date", "stock_name", "net_score"]].to_excel("FinalScores_"+today+'.xlsx')
    send_pdf_telegram("FinalScores_"+today+".xlsx", "لیست تمام سهم های مورد بررسی به ترتیب امتیاز موقعیت خرید برای افق ۳۰ روزه از دید خرد جمعی مدل های یادگیری عمیق", "417161976")
    send_pdf_telegram ("FinalScores_"+today+".xlsx", "لیست تمام سهم های مورد بررسی به ترتیب امتیاز موقعیت خرید برای افق ۳۰ روزه از دید خرد جمعی مدل های یادگیری عمیق", "-1001468757086")
    send_pdf_telegram ("FinalScores_"+today+".xlsx", "لیست تمام سهم های مورد بررسی به ترتیب امتیاز موقعیت خرید برای افق ۳۰ روزه از دید خرد جمعی مدل های یادگیری عمیق", "626243872")

    os.remove("FinalScores_"+today+".xlsx")