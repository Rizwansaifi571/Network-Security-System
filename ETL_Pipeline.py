import os 
import sys 
import json 
import pandas as pd
import numpy as np
import pymongo
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging

from dotenv import load_dotenv
load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
# print(MONGO_DB_URL)

class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def csv_to_json_convertor(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop = True, inplace = True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def insert_data_mongodb(self, records, database, collection):
        try:
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]

            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return (len(self.records))
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
if __name__ == "__main__":
    File_path = "Data\phisingData.csv"
    Database = "Data0"
    Collection = "NetworkData"
    
    obj1 = NetworkDataExtract()
    records = obj1.csv_to_json_convertor(File_path)
    print(records)
    no_of_records = obj1.insert_data_mongodb(records, Database, Collection)
    print(no_of_records)