from feature_extraction.feature_extraction import stock_daily_feature_extraction
from data_cleansing.daily_data_cleansing import CLEANED_DATA_FILENAME as DAILY_CLEANED_DATA_FILENAME
from data_cleansing.daily_data_cleansing import CLEANED_DATA_SUFFIX as DAILY_CLEANED_DATA_SUFFIX
from data_cleansing.index_data_cleansing import CLEANED_DATA_FILENAME as INDEX_CLEANED_DATA_FILENAME
from data_cleansing.index_data_cleansing import CLEANED_DATA_SUFFIX as INDEX_CLEANED_DATA_SUFFIX
from all_market_daily_denoised_data import DAILY_DENOISED_DATA_FILENAME, DAILY_DENOISED_INDEX_FILENAME
from all_market_daily_denoised_data import DENOISE

clean_daily_data_directory = 'data/clean_data'
clean_index_data_directory = 'data/clean_data'
saving_directory = 'data/feature_data'
group_info_directory = 'configs'
group_info_filename = 'stock_ids'
group_info_suffix = '.csv'

if DENOISE=='True':
    print('--------feature extraction using denoised data--------')
    clean_daily_data_filename = DAILY_DENOISED_DATA_FILENAME
    clean_daily_data_suffix = group_info_suffix
    clean_index_data_filename = DAILY_DENOISED_INDEX_FILENAME
    clean_index_data_suffix = group_info_suffix
else:
    print('--------feature extraction using noisy data--------')
    clean_daily_data_filename = DAILY_CLEANED_DATA_FILENAME
    clean_daily_data_suffix = DAILY_CLEANED_DATA_SUFFIX
    clean_index_data_filename = INDEX_CLEANED_DATA_FILENAME
    clean_index_data_suffix = INDEX_CLEANED_DATA_SUFFIX

#####################################
report_dict = stock_daily_feature_extraction(saving_directory, clean_daily_data_directory, clean_index_data_directory, \
    group_info_directory,\
    group_info_filename = group_info_filename,group_info_suffix = group_info_suffix ,\
    clean_daily_data_filename=clean_daily_data_filename, clean_daily_data_suffix=clean_daily_data_suffix, \
    clean_index_data_filename=clean_index_data_filename, clean_index_data_suffix=clean_index_data_suffix)

print('Finishing!')