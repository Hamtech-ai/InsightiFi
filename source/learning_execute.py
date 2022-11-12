from classifiers.classifier_manager import all_execute
from classifiers.data_splitting import data_splitting

from pandas import DataFrame
import pandas as pd
import sys, getopt

feature_data_file = 'data/feature_data/daily_feature_data.csv'
feature_detail_file = 'data/feature_data/daily_feature_detail.json'
test_train_saving_folder = 'data/splitting_data'
stocks_config_file = 'configs/stock_ids.csv'
best_stocks_file = 'configs/best_stocks.json'
extra_stocks_file = 'configs/eval_stocks.json'
starting_date = pd.Timestamp('2010-04-01') # if None, the beggining of the time
test_split_proportion = 0.3

test_train_folder = 'data/splitting_data'
saving_folder = 'data/result_data'
model_saving_folder = 'data/learned_model'
saving_prefix = 'beststock'

test_train_file = test_train_folder + '/' + saving_prefix + '_' + 'TestTrainSplitted.json'

#####################################
if __name__ == "__main__":
    try:
        argv = sys.argv[1:]
        opts, args = getopt.getopt(argv,"h:e",["horizon=", "evaluate"])
    except getopt.GetoptError:
        print('Wrong arguments')
        sys.exit(2)
    
    # arguments
    horizon = None
    eval_mode = False
    for opt, arg in opts:
        if opt in ("-h", "--horizon"):
            horizon = arg
        if opt in ("-e", "--evaluate"):
            eval_mode = True

    # steps
    data_splitting(feature_data_file, feature_detail_file, test_train_saving_folder, best_stocks_file, stocks_config_file, \
        starting_date=starting_date, test_split_proportion=test_split_proportion, 
        eval_stocks_file=extra_stocks_file, horizon=horizon, saving_prefix=saving_prefix)
    all_execute(feature_data_file, test_train_file, saving_folder, model_saving_folder, \
        horizon=horizon, saving_prefix=saving_prefix, eval_mode=eval_mode)

    print('Finishing!')