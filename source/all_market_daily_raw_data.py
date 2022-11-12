from data_fetching.daily_data_fetch import market_daily_data_save as market_daily_data_save

import pandas as pd
import datetime as datetime
import sys
import time as time

stock_id_file = 'configs/stock_ids.csv'
saving_directory = 'data/raw_data'
start_date = None
end_date = None
number_of_try = 100

###################################
symbol_id_input_data = pd.read_csv(stock_id_file, dtype={'id':str})
ids = [elem for elem in symbol_id_input_data['id']]
names = [str(elem) for elem in symbol_id_input_data['symbol']] 
id_name_dict = {id:name for id,name in zip(ids,names)}

# turn off warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

#####################################
start_time = time.time()

for try_num in range(number_of_try):
    # logging
    print('--------------- Fetching epoch %d ---------------'%(try_num + 1))
    report_dict = market_daily_data_save(ids, names, start_date, end_date, saving_directory)

    # error handling
    print('Errors:')
    print(report_dict)

    # handling fetching errors
    if len(report_dict['fetching_error']) > 0:
        id_name_dict = {key:id_name_dict[key] for key in list(report_dict['fetching_error'].keys())}

        ids = [key for key,val in id_name_dict.items()]
        names = [val for key,val in id_name_dict.items()]
    else:
        break

executation_time = time.time() - start_time

print('executation time: %f min' % (executation_time/60.0))
print('Finishing!')