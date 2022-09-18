import pandas as pd
import numpy as np
import os as os
import datetime as dt
import json as json
import psycopg2
import pandas.io.sql as sqlio

conn    =  psycopg2.connect(host='37.152.186.128', port = 5432, database='aftermarket', user='postgres', password='admin123!@#')
cursor  =  conn.cursor()
sql     = sql = "select date, stock_name, buy_score, sell_score, buy_score - sell_score as net_score \
from ml_scores where (buy_score) >0.6"

conn.commit()
window = sqlio.read_sql_query(sql, conn)
print(window)
window.to_csv("top_buy_scores.csv")