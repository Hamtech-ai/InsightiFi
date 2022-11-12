import pandas as pd
import numpy as np
import os as os
import datetime as dt
import json as json
import psycopg2
import requests
import re
import pandas.io.sql as sqlio
import kaleido
import plotly.graph_objects as go
from persiantools.jdatetime import JalaliDate
def _multiple_replace(mapping, text):
    """
    Internal function for replace all mapping keys for a input string
    :param mapping: replacing mapping keys
    :param text: user input string
    :return: New string with converted mapping keys to values
    """
    pattern = "|".join(map(re.escape, mapping.keys()))
    return re.sub(pattern, lambda m: mapping[m.group()], str(text))
def convert_en_numbers(input_str):
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
def send_image_telegram (image, caption, chat_id):
    from requests_toolbelt import MultipartEncoder
    fields = {"bot-name": "hermes", "chat-id" :chat_id, "image":( image, open(image, "rb"), 'image/jpeg'),
             "parse-mode": "html", "caption":caption}
    url = 'http://159.223.11.86:8891/send-image'
    m = MultipartEncoder(fields, boundary= 'my_super_custom_header')
    r = requests.post(url, headers = {'Content-Type': m.content_type}, data = m.to_string())

today = dt.datetime.today().strftime('%Y%m%d')  
cwd = os.getcwd()
sign_location = cwd + '/data'+'/Predictions' + '/Ensemble_' + 'Sign' + '_Signal' + today +'.csv'


df_sign = pd.read_csv(sign_location)

today_date = df_sign.date.max()
df_sign = df_sign[df_sign['date']==today_date]
df_sign = df_sign.drop(['Label', 'Unnamed: 0'],axis=1)
df_sign['Score'] = df_sign['Score'].round(2)
df_sign = df_sign.sort_values(by='Score', ascending = False)
df_sign = df_sign.reset_index(drop = True)
df_sign.rename  (columns = {'Score':'sign_score'}, inplace = True)


conn =  psycopg2.connect(host='37.152.186.128', port = 5432, database='aftermarket', \
                               user='postgres', password='admin123!@#')
cursor = conn.cursor()
#cursor.executemany("insert into market_scores(date, stock_name, sign_scores) values (%s, %s, %s)" , df_sign.values.tolist() )
#conn.commit()
sql = "select * from ml_scores where DATE(date) = DATE (NOW())"
conn.commit()
ml_scores = sqlio.read_sql_query(sql, conn)
ml_scores["date"] = ml_scores["date"].astype(str)
conn.close()
all_scores = df_sign.merge(ml_scores, on = ["stock_name","date"])
all_scores["sign_score"] = (all_scores["sign_score"] * 2) - 1
all_scores["final_score"] = all_scores["buy_score"] - all_scores["sell_score"] +all_scores["sign_score"]


conn    =  psycopg2.connect(host='193.176.242.204', port = 5432, database='postgres', user='kimia', password='hello')
cursor  =  conn.cursor()
sql     =  "select * from marketstats"
conn.commit()
MarketStats = sqlio.read_sql_query(sql, conn)
MarketStats.rename(columns = {'symbol':'stock_name'}, inplace = True)
conn.close()

all_scores = all_scores.merge(MarketStats[["stock_name", "market_cap"]], on = "stock_name",  how='left')
all_scores = all_scores[~all_scores.market_cap.isnull()].reset_index(drop=True)
all_scores['weight'] = all_scores['market_cap']/all_scores['market_cap'].sum()
all_scores['index_maker'] = all_scores['weight']*all_scores['final_score']

market_timing = int(100*all_scores['index_maker'].sum()/2)
market_timing_df = pd.DataFrame(columns = ["date", "market_timing"])
market_timing_df.loc[len(market_timing_df)] = [today, market_timing]

conn    =  psycopg2.connect(host='37.152.186.128', port = 5432, database='aftermarket', user='postgres', password='admin123!@#')
cursor  =  conn.cursor()
sql     =  "select * from market_timing order by date desc limit 1"
conn.commit()
yesterday_score = sqlio.read_sql_query(sql , conn).market_timing[0]
conn.close()


conn =  psycopg2.connect(host='37.152.186.128', port = 5432, database='aftermarket', \
                               user='postgres', password='admin123!@#')
cursor = conn.cursor()
cursor.execute("insert into market_timing (date, market_timing) values (%s, %s)" , market_timing_df.values.tolist()[0] )
conn.commit()
conn.close()






if market_timing<-50: color = 'rgb(228,26,28)'
elif market_timing>=-50 and market_timing<0: color = 'rgb(249,123,114)'
elif market_timing>=0 and market_timing<50: color = 'rgb(166,216,84)'
elif market_timing>=50: color = 'rgb(17,119,51)'
fig = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = market_timing,
    title = {'text': 'امتیاز بازار در سی روز آینده '},
    delta = {'reference': yesterday_score},
    gauge = {
        'axis':{'range': [-100, 100], 'dtick':10},
        'bar': {'color': color},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [-100, -50], 'color': 'rgba(228,26,28,0.4)', 'name':'فروش'},
            {'range': [-50, 0],  'color': 'rgba(249,123,114,0.2)', 'name':'نسبتا منفی'},
            {'range': [0,50],    'color': 'rgba(166,216,84,0.2)',  'name':'نسبتا مثبت'},
            {'range': [50,100],    'color': 'rgba(17,119,51,0.3)', 'name':'خرید'}],
    }
))   
fig.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})
fig.show()
fig.write_image("fig1.png")


today = dt.date.today()
today_farsi = convert_en_numbers(JalaliDate(today).strftime("%Y-%m-%d"))


send_image_telegram("fig1.png", "مارکت تایمینگ برای"+ today_farsi , "417161976" )
send_image_telegram("fig1.png", "مارکت تایمینگ برای"+ today_farsi , "-1001468757086" )

os.remove("fig1.png")

print(all_scores)