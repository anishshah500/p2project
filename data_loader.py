import os
import sqlite3
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

from pymongo import MongoClient
from datetime import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)

start_date_limit, end_date_limit = '1995-01-01', '2024-08-09'

class DataLoader():
    """
    A class to collect and store data from Yahoo Finance.
    """

    def __init__(self, start_date, end_date):
        """
        Initialize the DataCollector with a list of tickers, and start and end dates.
        """
        self.indices = ['^GSPC', '^RUT', '^IXIC']  # S&P 500, Russell 2000, Nasdaq 100
        self.start_date = start_date
        self.end_date = end_date

    def get_db(self):
        """
        Get a connection to the MongoDB database.
        """
        
        # Get MongoDB URI from environment variable
        mongo_uri = "mongodb+srv://user1:AhsyWdXCubfjdaFT@cluster0.kt9eu.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        if mongo_uri:
            print("MONGODB_URI: " + mongo_uri)
            
            client = MongoClient(mongo_uri, connect=False)
            db = client['equities_db']
            return db
        else:
            print("Check connection.....")
            return None

    def store_data(self, data):
        """
        Store the data in the MongoDB database.
        """
        db = self.get_db()
        try:
            if len(data):
                data = pd.DataFrame(data.reset_index())
                data['Date'] = data['Date'].astype(str)
                records = data.to_dict(orient='records')
                db.equities.insert_many(records)
        except Exception as e:
            print("Failed...")
            print(e)

    def retrieve_data(self, start_date, end_date):
        """
        Retrieve the data from the MongoDB database.
        """
        db = self.get_db()

        # Create the query
        query = {
            "Date": {
                "$gte": datetime.strftime(start_date, "%Y-%m-%d"),
                "$lt": datetime.strftime(end_date, "%Y-%m-%d")
            }
        }

        df = pd.DataFrame(db.equities.find(query)).iloc[:,1:]
        return df

    def get_tickers(self):
        """
        Get the list of tickers.
        """
        return pd.read_csv("all_tickers.csv")["Ticker"].tolist()

    def download_data(self):
        """
        Download the data from Yahoo Finance.
        """

        # If database exists delete it and re-create
        try:
            db.validate_collection("equities")  
            self.get_db().equities.drop()
        except Exception as e:
            print(e)
            print("This collection doesn't exist, something didn't really work")

        tickers1 = self.get_tickers()
        data = yf.download(tickers1, start=start_date_limit, end=end_date_limit)['Adj Close']
        data = data.dropna(axis=1, how='all')
        self.store_data(data)
        
    def get_equities_data(self, start_date, end_date, force_refresh=False):
        """
        Check if the data is already in the database, and if not, download it.
        If force_refresh is True, download the data even if it is already in the database.
        """
        if force_refresh:
            self.download_data()
            return self.retrieve_data(start_date, end_date)
        else:
            try:
                return self.retrieve_data(start_date, end_date)
            except Exception as e:
                self.download_data()
                return self.retrieve_data(start_date, end_date)