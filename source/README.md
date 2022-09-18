**This file contains all steps for codes that you should run daily or any time that you make changes in your repository.**

---
The explanation for how you can work with repository goes below:
## Edit a file

You’ll start by editing this README file to learn how to edit a file in Bitbucket.

1. Click **Source** on the left side.
2. Click the README.md link from the list of files.
3. Click the **Edit** button.
4. Delete the following text: *Delete this line to make a change to the README from Bitbucket.*
5. After making your change, click **Commit** and then **Commit** again in the dialog. The commit page will open and you’ll see the change you just made.
6. Go back to the **Source** page.

---

## Create a file

Next, you’ll add a new file to this repository.

1. Click the **New file** button at the top of the **Source** page.
2. Give the file a filename of **contributors.txt**.
3. Enter your name in the empty file space.
4. Click **Commit** and then **Commit** again in the dialog.
5. Go back to the **Source** page.

Before you move on, go ahead and explore the repository. You've already seen the **Source** page, but check out the **Commits**, **Branches**, and **Settings** pages.

---

## Clone a repository

Use these steps to clone from SourceTree, our client for using the repository command-line free. Cloning allows you to work on your files locally. If you don't yet have SourceTree, [download and install first](https://www.sourcetreeapp.com/). If you prefer to clone from the command line, see [Clone a repository](https://confluence.atlassian.com/x/4whODQ).

1. You’ll see the clone button under the **Source** heading. Click that button.
2. Now click **Check out in SourceTree**. You may need to create a SourceTree account or log in.
3. When you see the **Clone New** dialog in SourceTree, update the destination path and name if you’d like to and then click **Clone**.
4. Open the directory you just created to see your repository’s files.

Now that you're more familiar with your Bitbucket repository, go ahead and add a new file locally. You can [push your change back to Bitbucket with SourceTree](https://confluence.atlassian.com/x/iqyBMg), or you can [add, commit,](https://confluence.atlassian.com/x/8QhODQ) and [push from the command line](https://confluence.atlassian.com/x/NQ0zDQ).

## NOTE: Pay attention to your .gitignore file and make folders that are needed in your repository.

### Run

# incremental_process.sh:
You should run this every working-day. This shell contains all the codes that you need for stock market prediction. 
NOTE: Shells run all the codes one after another and a bug in one .py code wouldn't stop the shell from running others. Thus, for debugging, you should also look backward and see whether there was some problem in former .py files :D

This shell includes:
## data preparation
all_market_daily_raw_data.py : **All data from first day of stock market till former working day**
all_market_incremental_data.py : **Data for current working day**
market_index_data.py : **index data !**

**saving folder : data/raw_data/...csv**

market_index_clean_data.py : **Cleaning index using conditional terms for obviouly wrong data**
all_market_daily_clean_data.py : **Like what we had for index**
addtoday_clean_data.py: **Sometimes data being fetch using all_market_incremental_data.py doesn't contain Ind and NonInd data. By running this code after your cleansing process, your current day data for Ind/NonInd would be added to your cleaned data**

**saving folder : data/clean_data/...csv**

## denoising
python all_market_daily_denoised_data.py : **This code asks you whether you want your data to be denoised or not. Remember that while you are denoising your regime for testing data would be recent 265 historical days and other regimes are not available for denoising.**
**saving folder in case denoising is set to True: data/clean_data/denoised_data....csv**

## feature extraction
python all_market_daily_feature_extraction.py : **Feature extraction for stock market. For furthur info you can using debug mode and see all pipe lines that are used in extracting features**
python all_market_overall_feature_extraction.py : **Features for market timing which are some selected features from stock selection part that their wheighted mean has been calculated.**

**saving folder : data/feature_data/...csv**
NOTE: You can also see the explanation about the daily feature in a json file available in the saving folder.

## learning
NOTE: There is a stock_selection_params.json file in your "configs" folder. This file contain label binary mapping, primary selected feature columns and your Random Forest Parameters which are for all kind of signals e.g buy, sell, side and all horizons e.g. 2, 7 etc. If you are trying to add a new model, you can add its parameters in this json file for better review. 

python TestTrainSplit.py :**Splitting data to train, test and evaluation data. Our test data would be the recent year of the market in this splitting approach, make labels using predefined thresholds** 

**saving folder: TestTrainData/TestTrainSplitted.json**

python stock_selection_memory_buy.py
python stock_selection_memory_sell.py
**Fitting models for earlier horizons, 2, 7 and 14 days forward return, for giving buy/sell signals**
**saving folder: predictions in data/PredictionFeatures/signal_horizon.csv. Also models would be saved in saved_models/signal_model_horizon.pkl**
NOTE: You don't need to run both of aformentioned .py files. In fact, it is necessary every time you make a change in your models or features as well as the first time that you are running this repository.

python model_export_buy.py
python model_export_sell.py
**Using models that are saved in the previous step and saved your updated predictions as your prediction features. Selected regime would be the one that you have chosen in your denoising process!**
**saving folder: data/BuyorSell_selected_features.csv**

python StockSelection30d_buy.py
python StockSelection30d_sell.py
**Models for final preictions**
**saving folder: data/Predictions/...csv and excel**

python MarketTiming.py 
**Markettimng model**
**saving folder: data/Predictions/...csv and excel**