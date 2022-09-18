from pandas.io import pytables
from stock_selection.stock_selection import stock_selection_final
from all_market_daily_denoised_data import REGIME as regime
from stock_selection.final_results_export import result_export

# inputs
test_train_file = 'TestTrainData' + '/' + 'TestTrainSplitted' + '.json'
model_horizon = '30d'
signal = 'Buy'
stock_selection_params_file = 'configs' + '/stock_selection_params' + '.json'

stock_selection_final(test_train_file, model_horizon, signal, stock_selection_params_file, regime)

result_export(model_horizon, signal)