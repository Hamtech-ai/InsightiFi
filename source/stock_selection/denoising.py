import pandas as pd
import numpy as np
import pywt
import copy


def lowpassfilter(signal, thresh, wavelet):
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    bg = reconstructed_signal.shape[0] - signal.shape[0]
    reconstructed_signal = reconstructed_signal[bg:]
    return reconstructed_signal

def smoothing_func_stocks_wavelet(stock_group):
    print('>', end='')
    wavelet = 'db4'
    low_pass_filter_threshold = 0.2
    df = stock_group
    initial_index = df.index
    df.index = [pd.Timestamp(item) for item in df['date']]
    columns_to_smooth = ['max_price', 'min_price', 'close_price', 'last_price',
                         'first_price', 'yesterday_price', 'value', 
                         'Individual_buy_value', 'NonIndividual_buy_value',
                         'Individual_sell_value', 'NonIndividual_sell_value',
                         'adj_max_price', 'adj_min_price', 'adj_first_price',
                         'adj_last_price', 'adj_close_price']
    columns_not_to_smooth = ['volume', 'count', 'Individual_buy_count', 'NonIndividual_buy_count',
                             'Individual_sell_count', 'NonIndividual_sell_count',
                             'Individual_buy_volume', 'NonIndividual_buy_volume',
                             'Individual_sell_volume', 'NonIndividual_sell_volume', 'adj_volume',
                             'date', 'stock_name']
    return_df = pd.DataFrame()
    
    if df.shape[0]<= 100:
        for col in columns_to_smooth+columns_not_to_smooth:
            return_df[col] = [np.nan]*df.shape[0]
    else:
        for col in columns_to_smooth:
            rec = lowpassfilter(df[col], low_pass_filter_threshold, wavelet)
            rec[0:40] = [np.nan]*40
            rec[-40:] = df[col].to_list()[-40:]
            return_df[col] = rec
        for col in columns_not_to_smooth:
            return_df[col] = df[col].to_list()
    return_df.index = initial_index
    return return_df
def smoothing_func_index_wavelet(index_data):
    wavelet = 'db4'
    low_pass_filter_threshold = 0.2

    rec = lowpassfilter(index_data['total_stock_index'], low_pass_filter_threshold, wavelet)
    rec[0:40] = [np.nan]*40
    rec[-40:] = index_data['total_stock_index'].to_list()[-40:]
    index_data['total_stock_index'] = rec

    return index_data

def smoothing_func_stocks_wma(stock_group):
    print('>', end='')
    df = stock_group
    initial_index = df.index
    df.index = [pd.Timestamp(item) for item in df['date']]
    columns_to_smooth = ['max_price', 'min_price', 'close_price', 'last_price',
                         'first_price', 'yesterday_price', 'value', 
                         'Individual_buy_value', 'NonIndividual_buy_value',
                         'Individual_sell_value', 'NonIndividual_sell_value',
                         'adj_max_price', 'adj_min_price', 'adj_first_price',
                         'adj_last_price', 'adj_close_price', 'volume', 'count',
                         'Individual_buy_count', 'NonIndividual_buy_count',
                         'Individual_sell_count', 'NonIndividual_sell_count',
                         'Individual_buy_volume', 'NonIndividual_buy_volume',
                         'Individual_sell_volume', 'NonIndividual_sell_volume', 'adj_volume']
    columns_not_to_smooth = ['date', 'stock_name']
    return_df = pd.DataFrame()
    
    for col in columns_to_smooth:
        df['smooth'] = ((0.5*df[col])+(0.25*df[col].shift(+1))+
                        (0.2*df[col].shift(+2))+(0.05*df[col].shift(+3)))
        return_df[col] = df['smooth']
    for col in columns_not_to_smooth:
        return_df[col] = df[col]
    return_df.index = initial_index

    return return_df
def smoothing_func_index_wma(index_data):
    
    index_data['total_stock_index'] =\
    ((0.5*index_data.total_stock_index)+(0.25*index_data.total_stock_index.shift(+1))+
     (0.2*index_data.total_stock_index.shift(+2))+(0.05*index_data.total_stock_index.shift(+3)))

    return index_data

def one_year_clean_split(Data, Index):

    Data = Data.reset_index(drop=True)
    Data = Data.drop(['Unnamed: 0'],axis=1)
    Data = Data.replace([np.inf, -np.inf], np.nan)
    Data['date'] = Data['date'].astype('datetime64[ns]')
    ending_date = pd.Series([max(Data['date'])])
    test_duration = pd.Series([pd.Timedelta('365 days')])
    eval_duration = pd.Series([pd.Timedelta('30 days')])
    evaluation_start_date = ending_date - eval_duration
    test_start_date = evaluation_start_date - test_duration
    df_new = copy.copy(Data)

    # splitting
    TrainData = df_new[df_new['date'] < str(test_start_date[0])]
    TestData = df_new[df_new['date'] >= str(test_start_date[0])]
    TrainLabel = Index[Index['date'] < str(test_start_date[0])]
    TestLabel = Index[Index['date'] >= str(test_start_date[0])]


    return TrainData, TestData, TrainLabel, TestLabel

def denoising_steps(clean_daily_data_directory, DAILY_CLEANED_DATA_FILENAME, DAILY_CLEANED_DATA_SUFFIX, \
    INDEX_CLEANED_DATA_FILENAME, INDEX_CLEANED_DATA_SUFFIX,\
    DAILY_DENOISED_DATA_FILENAME,DAILY_DENOISED_INDEX_FILENAME,\
    DENOISE_SUFFIX, DENOISE, REGIME):    
    if ((DENOISE =='True') and (REGIME =='Current')):   
        # -----------------------------------------------------------------------------------
        stock_clean_data = pd.read_csv(clean_daily_data_directory+'/'+DAILY_CLEANED_DATA_FILENAME+DAILY_CLEANED_DATA_SUFFIX)
        index_clean_data = pd.read_csv(clean_daily_data_directory+'/'+INDEX_CLEANED_DATA_FILENAME+INDEX_CLEANED_DATA_SUFFIX)



        TrainData, TestData, TrainLabel, TestLabel = one_year_clean_split(stock_clean_data,index_clean_data)

        #-----------------------------------------------------
        TrainData.dropna(inplace=True)
        TrainData = TrainData.sort_values(by=['stock_name', 'date'], ascending=[True, True]).reset_index(drop=True)
        TrainGoups = TrainData.groupby('stock_name')
        smooth_train_data = TrainGoups.apply(smoothing_func_stocks_wavelet).reset_index(drop=True)

        TestData.dropna(inplace=True)
        TestData = TestData.sort_values(by=['stock_name', 'date'], ascending=[True, True]).reset_index(drop=True)
        # TestGroups = TestData.groupby('stock_name')
        # smooth_test_data = TestGroups.apply(smoothing_func_stocks_wavelet).reset_index(drop=True)

        smooth_data = smooth_train_data.append(TestData, ignore_index=True)
        smooth_data = smooth_data.sort_values(by=['stock_name', 'date'], ascending=[True, True]).reset_index(drop=True)
        #-----------------------------------------------------
        TrainLabel = smoothing_func_index_wavelet(TrainLabel)
        # TestLabel  = smoothing_func_index_wavelet(TestLabel)

        index_smooth_data = TrainLabel.append(TestLabel, ignore_index=True)
        index_smooth_data = index_smooth_data.sort_values(by=['date'], ascending=[True]).reset_index(drop=True)
        #-----------------------------------------------------


        # stock_clean_data.dropna(inplace=True)
        # stock_clean_data = stock_clean_data.\
        # sort_values(by=['stock_name', 'date'], ascending=[True, True]).reset_index(drop=True)

        # stock_groups = stock_clean_data.groupby('stock_name')
        # smooth_data = stock_groups.apply(smoothing_func_stocks_wma).reset_index(drop=True)

        # index_smooth_data = smoothing_func_index_wma(index_clean_data)

        smooth_data.to_csv(clean_daily_data_directory+'/'+DAILY_DENOISED_DATA_FILENAME+DENOISE_SUFFIX)
        index_smooth_data.to_csv(clean_daily_data_directory+'/'+DAILY_DENOISED_INDEX_FILENAME+DENOISE_SUFFIX)
        print('smooth data saved!')
    else :
        print('denoising skipped because of self choice or selecting a regime rather than current')    

    return
