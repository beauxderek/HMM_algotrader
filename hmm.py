import os
import pickle
from hmmlearn.hmm import GaussianHMM
import numpy as np
from data_handler import fetch_all_data
from datetime import datetime, time
from config import SECURITIES, START_DATE, TRAIN_END_DATE, END_DATE, N_COMPONENTS
import logging

logger = logging.getLogger(__name__)

def train_and_save_model(ticker, data, timeframe):
    """
    Train an HMM model on the returns and save it to a file.
    
    Parameters:
        ticker (str): The stock symbol
        data (pd.DataFrame): Historical price data
        timeframe (str): Data timeframe ('daily' or 'hourly')
    """
    if data.empty:
        logger.warning(f"No {timeframe} data for {ticker} between {START_DATE} and {TRAIN_END_DATE}.")
        return
    
    # Ensure we're working with the correct date range
    data = data.loc[START_DATE:TRAIN_END_DATE]
    
    # Use 'Returns' column instead of calculating it here
    rets = data['returns'].values.reshape(-1, 1)
    
    model = GaussianHMM(n_components=N_COMPONENTS, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(rets)
    
    logger.info(f"Model score for {ticker} ({timeframe} data): {model.score(rets)}")
    
    # Save the model with timeframe indicator
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    filename = f"{model_dir}/{ticker}_hmm_model.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    
    logger.info(f"Saved {timeframe} model for {ticker} to {filename}")

def main(api_choice, timeframe='daily'):
    """
    Train HMM models for all securities.
    
    Parameters:
        api_choice (str): The data source to use ('yfinance', 'polygon', or 'sqlite')
        timeframe (str): Data timeframe to use ('daily' or 'hourly')
    """
    logger.info(f"Fetching {timeframe} data using {api_choice}...")
    data = fetch_all_data(api_choice, START_DATE, TRAIN_END_DATE, timeframe)
    
    for ticker in SECURITIES:
        logger.info(f"Training model for {ticker} using {timeframe} data...")
        if ticker in data:
            train_and_save_model(ticker, data[ticker], timeframe)
        else:
            logger.warning(f"No {timeframe} data available for {ticker}")

if __name__ == "__main__":
    import sys
    api_choice = sys.argv[1] if len(sys.argv) > 1 else 'yfinance'
    timeframe = sys.argv[2] if len(sys.argv) > 2 else 'daily'
    main(api_choice, timeframe)