from stock_selection.denoising import denoising_steps
from data_cleansing.daily_data_cleansing import CLEANED_DATA_FILENAME as DAILY_CLEANED_DATA_FILENAME
from data_cleansing.daily_data_cleansing import CLEANED_DATA_SUFFIX as DAILY_CLEANED_DATA_SUFFIX
from data_cleansing.index_data_cleansing import CLEANED_DATA_FILENAME as INDEX_CLEANED_DATA_FILENAME
from data_cleansing.index_data_cleansing import CLEANED_DATA_SUFFIX as INDEX_CLEANED_DATA_SUFFIX

import pandas as pd
import numpy as np
import copy 

clean_daily_data_directory = 'data/clean_data'
clean_daily_data_filename = DAILY_CLEANED_DATA_FILENAME
clean_daily_data_suffix = DAILY_CLEANED_DATA_SUFFIX
clean_index_data_directory = 'data/clean_data'
clean_index_data_filename = INDEX_CLEANED_DATA_FILENAME
clean_index_data_suffix = INDEX_CLEANED_DATA_SUFFIX
saving_directory = 'data/clean_data'
DAILY_DENOISED_DATA_FILENAME = 'denoised_data_stocks'
DAILY_DENOISED_INDEX_FILENAME = 'denoised_data_index'
DENOISE_SUFFIX = '.csv'
REGIME = 'Sell'
DENOISE = 'False'
    
denoising_steps(clean_daily_data_directory, DAILY_CLEANED_DATA_FILENAME, DAILY_CLEANED_DATA_SUFFIX, \
    INDEX_CLEANED_DATA_FILENAME, INDEX_CLEANED_DATA_SUFFIX,\
    DAILY_DENOISED_DATA_FILENAME,DAILY_DENOISED_INDEX_FILENAME,\
    DENOISE_SUFFIX, DENOISE, REGIME)