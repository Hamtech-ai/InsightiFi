# data preparation
#python all_market_daily_raw_data.py
python all_market_incremental_data.py
python market_index_data.py
python market_index_clean_data.py
python all_market_daily_clean_data.py
python addtoday_clean_data.py
python all_market_daily_denoised_data.py

# feature extraction
python all_market_daily_feature_extraction.py

# learning
python TestTrainSplit.py
python stock_selection_memory_buy.py
python model_export_buy.py
python StockSelection30d_buy.py
python stock_selection_memory_sell.py
python model_export_sell.py
python StockSelection30d_sell.py
#python stock_selection_final_scores.py

# SIGN
python stock_selection_memory_sign.py
python model_export_sign.py
python StockSelection30d_sign.py
#python MarketTiming_todb.py
