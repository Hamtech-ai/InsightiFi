from data_fetching.index_data_fetch import RAW_DATA_SUFFIX as raw_data_suffix
from data_fetching.index_data_fetch import RAW_DATA_PREFIX as raw_data_prefix
from data_cleansing.index_data_cleansing import index_data_cleansing

raw_data_directory = 'data/raw_data'
raw_data_prefix = raw_data_prefix
raw_data_suffix = raw_data_suffix
saving_directory = 'data/clean_data'

#####################################
report_dict = index_data_cleansing(saving_directory, raw_data_directory, raw_data_prefix=raw_data_prefix, raw_data_suffix=raw_data_suffix)

print('Finishing!')