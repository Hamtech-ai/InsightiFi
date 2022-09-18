from data_fetching.index_data_fetch import market_index_data_save as market_index_data_save

import pandas as pd
import datetime as datetime

index_id_file = 'configs/index_ids.csv'
saving_directory = 'data/raw_data'
start_date = None
end_date = None
number_of_try = 10

###################################
index_id_input_data = pd.read_csv(index_id_file)
ids = [str(elem) for elem in index_id_input_data['id']]
names = [str(elem) for elem in index_id_input_data['index']]
id_name_dict = {id:name for id,name in zip(ids,names)}

###################################
for try_num in range(number_of_try):
    # logging
    print('--------------- Fetching epoch %d ---------------'%(try_num + 1))
    report_dict = market_index_data_save(ids, names, start_date, end_date, saving_directory)

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

print('Finishing!')