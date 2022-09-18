from data_fetching.daily_data_fetch import RAW_DATA_SUFFIX as raw_data_suffix
from data_fetching.daily_data_fetch import RAW_DATA_PREFIX as raw_data_prefix
from data_cleansing.daily_data_cleansing import daily_data_cleansing

import pandas as pd

stock_id_file = 'configs/stock_ids.csv'
raw_data_directory = 'data/raw_data'
raw_data_prefix = raw_data_prefix
raw_data_suffix = raw_data_suffix
saving_directory = 'data/clean_data'

###################################
symbol_id_input_data = pd.read_csv(stock_id_file)
ids = [str(elem) for elem in symbol_id_input_data['id']]
names = [str(elem) for elem in symbol_id_input_data['symbol']] 
id_name_dict = {id:name for id,name in zip(ids,names)}

#####################################
report_dict = daily_data_cleansing(saving_directory, raw_data_directory, \
    raw_data_prefix=raw_data_prefix, raw_data_suffix=raw_data_suffix, names=names)

print('Finishing!')