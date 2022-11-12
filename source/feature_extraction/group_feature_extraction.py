import pandas as pd
import numpy as np
import datetime as datetime
import jdatetime as jdatetime
import swifter as swifter

# TODO: merge data with index
industry_columns = ['متانول', 'هتل و رستوران',
       'کاشی و سرامیک', 'زراعت و دامپروری', 'هلدینگ', 'سرمایه گذاری',
       'پلاستیک', 'پتروشیمی', 'محصولات کاغذی', 'ماشین آلات و تحهیزات', 'غذایی',
       'ماشین آلات و تجهیزات برقی', 'خدمات مهندسی و پیمانکاری', 'سنگ آهنی',
       'آلومینیومی', 'مس و منسوجات مسی', 'فرآورده های فولادی', 'نیروگاهی',
       'فولادی', 'پالایشی', 'سرب و روی', 'مخابرات', 'البسه و منسوجات',
       'قند و شکر', 'ذغال سنگ', 'روانکار', 'اوره', 'کانه غیر فلزی', 'دارو',
       'پخش و خرده فروشی', 'سیمان', 'شوینده', 'بانک', 'بیمه', 'آی تی',
       'حمل و نقل', 'خودرو قطعات', 'فراکاب', 'تامین سرمایه ', 'ساختمان',
       'لیزینگ', 'روغنی', 'لوازم خانگی', 'خدمات پرداخت', 'تولیدی','دلاری']
dollar_based_columns = "دلاری"
other_groups = ['متانول', 'هتل و رستوران',
       'کاشی و سرامیک', 'زراعت و دامپروری', 'هلدینگ', 'سرمایه گذاری',
       'پلاستیک', 'پتروشیمی', 'محصولات کاغذی', 'ماشین آلات و تحهیزات', 'غذایی',
       'ماشین آلات و تجهیزات برقی', 'خدمات مهندسی و پیمانکاری', 'سنگ آهنی',
       'آلومینیومی', 'مس و منسوجات مسی', 'فرآورده های فولادی', 'نیروگاهی',
       'فولادی', 'پالایشی', 'سرب و روی', 'مخابرات', 'البسه و منسوجات',
       'قند و شکر', 'ذغال سنگ', 'روانکار', 'اوره', 'کانه غیر فلزی', 'دارو',
       'پخش و خرده فروشی', 'سیمان', 'شوینده', 'بانک', 'بیمه', 'آی تی',
       'حمل و نقل', 'خودرو قطعات', 'فراکاب', 'تامین سرمایه ', 'ساختمان',
       'لیزینگ', 'روغنی', 'لوازم خانگی', 'خدمات پرداخت', 'تولیدی']

def group_based_features(stock_daily_data_with_groups):
    print('industry group based features')

    # industry indication
    # Assumption: only the first industry in the list is considered
    stock_groups = np.array(stock_daily_data_with_groups[industry_columns])
    group_labels = np.argmax(stock_groups, axis=1)
    group_labels = pd.DataFrame(data=group_labels, columns=["group_labels"])
    temp = stock_daily_data_with_groups.drop(industry_columns, axis=1)
    temp = temp.drop(['id', 'ISIN'], axis=1)
    df = pd.concat([temp, group_labels], axis=1)
    
    # result initialization
    feature_data = pd.DataFrame([])
    feature_detail = dict()

    # grouping
    grouped_df = df.groupby(["date", "group_labels"])

    # map function
    def bwd_group_mean_price_func(grouped_df):
        initial_index = grouped_df.index
        df = grouped_df

        group_close_price = np.mean(df['close_price'])
        group_yesterday_price = np.mean(df['yesterday_price'])
        return_df = pd.DataFrame({\
            'group_close_price':group_close_price, \
            'group_yesterday_price':group_yesterday_price, \
            }, index=initial_index)
        return_df.index = initial_index
        return return_df

    group_price_feature_data = grouped_df.apply(bwd_group_mean_price_func).reset_index(drop=True)

    group_price_feature_detail = {\
            'group_close_price':{'description':'daily mean of close price in industry groups.'}, \
            'group_yesterday_price':{'description':'daily mean of yesterday price in industry groups.'}, \
            }

    # relative group features
    cols_go_in_func = ['date', 'close_price', 'yesterday_price', 'stock_name']
    stock_with_group_price = pd.concat([df[cols_go_in_func], group_price_feature_data], axis=1)

    stock_groups = stock_with_group_price.groupby('stock_name')
    def bwd_group_ret_price_func(stock_groups):
        initial_index = stock_groups.index
        df = stock_groups
        df.index = [pd.Timestamp(item) for item in df['date']]

        grp_ret1d = (np.log(df['group_close_price']) - np.log(df['group_yesterday_price'])).rolling('1d').sum()
        grp_ret3d = (np.log(df['group_close_price']) - np.log(df['group_yesterday_price'])).rolling('3d').sum()
        grp_ret7d = (np.log(df['group_close_price']) - np.log(df['group_yesterday_price'])).rolling('7d').sum()
        grp_ret14d = (np.log(df['group_close_price']) - np.log(df['group_yesterday_price'])).rolling('14d').sum()
        grp_ret30d = (np.log(df['group_close_price']) - np.log(df['group_yesterday_price'])).rolling('30d').sum()
        grp_ret60d = (np.log(df['group_close_price']) - np.log(df['group_yesterday_price'])).rolling('60d').sum()
        grp_ret90d = (np.log(df['group_close_price']) - np.log(df['group_yesterday_price'])).rolling('90d').sum()
        grp_ret120d = (np.log(df['group_close_price']) - np.log(df['group_yesterday_price'])).rolling('120d').sum()
        grp_ret300d = (np.log(df['group_close_price']) - np.log(df['group_yesterday_price'])).rolling('300d').sum()
        
        full_ret1d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('1d').sum()
        full_ret3d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('3d').sum()
        full_ret7d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('7d').sum()
        full_ret14d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('14d').sum()
        full_ret30d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('30d').sum()
        full_ret60d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('60d').sum()
        full_ret90d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('90d').sum()
        full_ret120d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('120d').sum()
        full_ret300d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('300d').sum()
        
        return_df = pd.DataFrame({\
            'ret_wrt_grp_ind_1d_log'  : full_ret1d.values - grp_ret1d.values, \
            'ret_wrt_grp_ind_3d_log' : full_ret3d.values - grp_ret3d.values, \
            'ret_wrt_grp_ind_7d_log' : full_ret7d.values - grp_ret7d.values, \
            'ret_wrt_grp_ind_14d_log' : full_ret14d.values - grp_ret14d.values, \
            'ret_wrt_grp_ind_30d_log' : full_ret30d.values - grp_ret30d.values, \
            'ret_wrt_grp_ind_60d_log' : full_ret60d.values - grp_ret60d.values, \
            'ret_wrt_grp_ind_90d_log' : full_ret90d.values - grp_ret90d.values, \
            'ret_wrt_grp_ind_120d_log': full_ret120d.values - grp_ret120d.values, \
            'ret_wrt_grp_ind_300d_log': full_ret300d.values - grp_ret300d.values, \
            })
        return_df.index = initial_index
        return return_df

    feature_data = stock_groups.apply(bwd_group_ret_price_func).reset_index(drop=True)
    
    feature_detail = {\
            'ret_wrt_grp_ind_1d_log'  : {'description':'deviation of 1-day-return of stock from 1-day-return of its industry group.'}, \
            'ret_wrt_grp_ind_3d_log' : {'description':'deviation of 3-day-return of stock from 3-day-return of its industry group.'}, \
            'ret_wrt_grp_ind_7d_log' : {'description':'deviation of 7-day-return of stock from 7-day-return of its industry group.'}, \
            'ret_wrt_grp_ind_14d_log' : {'description':'deviation of 14-day-return of stock from 14-day-return of its industry group.'}, \
            'ret_wrt_grp_ind_30d_log' : {'description':'deviation of 30-day-return of stock from 30-day-return of its industry group.'}, \
            'ret_wrt_grp_ind_60d_log' : {'description':'deviation of 60-day-return of stock from 60-day-return of its industry group.'}, \
            'ret_wrt_grp_ind_90d_log' : {'description':'deviation of 90-day-return of stock from 90-day-return of its industry group.'}, \
            'ret_wrt_grp_ind_120d_log': {'description':'deviation of 120-day-return of stock from 120-day-return of its industry group.'}, \
            'ret_wrt_grp_ind_300d_log': {'description':'deviation of 300-day-return of stock from 120-day-return of its industry group.'}, \
            }

    return feature_data, feature_detail

def dollar_group_based_features(stock_daily_data_with_groups):
    print('dollar group based features')

    df = stock_daily_data_with_groups
    df = df.drop(other_groups, axis=1)
    df = df.drop(['id', 'ISIN'], axis=1)

    # date groups
    grouped_df = df.groupby(["date", dollar_based_columns])

    def bwd_group_mean_price_func(grouped_df):
        initial_index = grouped_df.index
        df = grouped_df

        dollar_close_price = np.mean(df['close_price'])
        dollar_yesterday_price = np.mean(df['yesterday_price'])
        return_df = pd.DataFrame({\
            'dollar_close_price':dollar_close_price, \
            'dollar_yesterday_price':dollar_yesterday_price, \
            }, index=initial_index)
        return_df.index = initial_index
        return return_df

    dollar_price_feature_data = grouped_df.apply(bwd_group_mean_price_func).reset_index(drop=True)

    dollar_price_feature_detail = {\
            'dollar_close_price':{'description':'daily mean of close price in dlr/non-dlr group.'}, \
            'dollar_yesterday_price':{'description':'daily mean of yesterday price in dlr/non-dlr group.'}, \
            }

    # relative group features
    cols_go_in_func = ['date','close_price','yesterday_price','stock_name']
    stock_with_group_price = pd.concat([df[cols_go_in_func],dollar_price_feature_data], axis=1)

    stock_groups = stock_with_group_price.groupby('stock_name')
    def bwd_group_ret_price_func(stock_groups):
        initial_index = stock_groups.index
        df = stock_groups
        df.index = [pd.Timestamp(item) for item in df['date']]
        
        dlr_ret1d = (np.log(df['dollar_close_price']) - np.log(df['dollar_yesterday_price'])).rolling('1d').sum()
        dlr_ret3d = (np.log(df['dollar_close_price']) - np.log(df['dollar_yesterday_price'])).rolling('3d').sum()
        dlr_ret7d = (np.log(df['dollar_close_price']) - np.log(df['dollar_yesterday_price'])).rolling('7d').sum()
        dlr_ret14d = (np.log(df['dollar_close_price']) - np.log(df['dollar_yesterday_price'])).rolling('14d').sum()
        dlr_ret30d = (np.log(df['dollar_close_price']) - np.log(df['dollar_yesterday_price'])).rolling('30d').sum()
        dlr_ret60d = (np.log(df['dollar_close_price']) - np.log(df['dollar_yesterday_price'])).rolling('60d').sum()
        dlr_ret90d = (np.log(df['dollar_close_price']) - np.log(df['dollar_yesterday_price'])).rolling('90d').sum()
        dlr_ret120d = (np.log(df['dollar_close_price']) - np.log(df['dollar_yesterday_price'])).rolling('120d').sum()
        dlr_ret300d = (np.log(df['dollar_close_price']) - np.log(df['dollar_yesterday_price'])).rolling('300d').sum()
        
        full_ret1d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('1d').sum()
        full_ret3d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('3d').sum()
        full_ret7d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('7d').sum()
        full_ret14d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('14d').sum()
        full_ret30d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('30d').sum()
        full_ret60d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('60d').sum()
        full_ret90d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('90d').sum()
        full_ret120d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('120d').sum()
        full_ret300d = (np.log(df['close_price']) - np.log(df['yesterday_price'])).rolling('300d').sum()
        
        return_df = pd.DataFrame({\
            'ret_wrt_dlr_ind_1d_log'  : full_ret1d.values - dlr_ret1d.values, \
            'ret_wrt_dlr_ind_3d_log' : full_ret3d.values - dlr_ret3d.values, \
            'ret_wrt_dlr_ind_7d_log' : full_ret7d.values - dlr_ret7d.values, \
            'ret_wrt_dlr_ind_14d_log' : full_ret14d.values - dlr_ret14d.values, \
            'ret_wrt_dlr_ind_30d_log' : full_ret30d.values - dlr_ret30d.values, \
            'ret_wrt_dlr_ind_60d_log' : full_ret60d.values - dlr_ret60d.values, \
            'ret_wrt_dlr_ind_90d_log' : full_ret90d.values - dlr_ret90d.values, \
            'ret_wrt_dlr_ind_120d_log': full_ret120d.values - dlr_ret120d.values, \
            'ret_wrt_dlr_ind_300d_log': full_ret300d.values - dlr_ret300d.values, \
            })
        return_df.index = initial_index
        return return_df

    feature_data = stock_groups.apply(bwd_group_ret_price_func).reset_index(drop=True)
    
    feature_detail = {\
            'ret_wrt_dlr_ind_1d_log'  : {'description':'deviation of 1-day-return of stock from 1-day-return of its dlr/non-dlr group.'}, \
            'ret_wrt_dlr_ind_3d_log' : {'description':'deviation of 3-day-return of stock from 3-day-return of its dlr/non-dlr group.'}, \
            'ret_wrt_dlr_ind_7d_log' : {'description':'deviation of 7-day-return of stock from 7-day-return of its dlr/non-dlr group.'}, \
            'ret_wrt_dlr_ind_14d_log' : {'description':'deviation of 14-day-return of stock from 14-day-return of its dlr/non-dlr group.'}, \
            'ret_wrt_dlr_ind_30d_log' : {'description':'deviation of 30-day-return of stock from 30-day-return of its dlr/non-dlr group.'}, \
            'ret_wrt_dlr_ind_60d_log' : {'description':'deviation of 60-day-return of stock from 60-day-return of its dlr/non-dlr group.'}, \
            'ret_wrt_dlr_ind_90d_log' : {'description':'deviation of 90-day-return of stock from 90-day-return of its dlr/non-dlr group.'}, \
            'ret_wrt_dlr_ind_120d_log': {'description':'deviation of 120-day-return of stock from 120-day-return of its dlr/non-dlr group.'}, \
            'ret_wrt_dlr_ind_300d_log': {'description':'deviation of 300-day-return of stock from 120-day-return of its dlr/non-dlr group.'}, \
            }

    return feature_data, feature_detail
