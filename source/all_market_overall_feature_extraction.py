from feature_extraction.feature_extraction import overal_feature_extraction
from feature_extraction.feature_extraction import FEATURE_DATA_FILENAME as FEATURE_DATA_FILENAME
from feature_extraction.feature_extraction import FEATURE_DATA_SUFFIX as FEATURE_DATA_SUFFIX
from data_cleansing.index_data_cleansing import CLEANED_DATA_FILENAME as INDEX_CLEANED_DATA_FILENAME
from data_cleansing.index_data_cleansing import CLEANED_DATA_SUFFIX as INDEX_CLEANED_DATA_SUFFIX

feature_daily_data_directory = 'data/feature_data'
feature_daily_data_filename = FEATURE_DATA_FILENAME
feature_daily_data_suffix = FEATURE_DATA_SUFFIX
clean_index_data_directory = 'data/clean_data'
clean_index_data_filename = INDEX_CLEANED_DATA_FILENAME
clean_index_data_suffix = INDEX_CLEANED_DATA_SUFFIX
saving_directory = 'data/feature_data'
group_info_directory = 'configs'
group_info_filename = 'stock_ids'
group_info_suffix = '.csv'

#####################################
report_dict = overal_feature_extraction(saving_directory, feature_daily_data_directory, clean_index_data_directory, \
    group_info_directory,\
    group_info_filename = group_info_filename,group_info_suffix = group_info_suffix ,\
    feature_daily_data_filename=feature_daily_data_filename, feature_daily_data_suffix=feature_daily_data_suffix, \
    clean_index_data_filename=clean_index_data_filename, clean_index_data_suffix=clean_index_data_suffix)

print('Finishing!')