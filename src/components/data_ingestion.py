import os 
import sys 
from src.logger import logging 
from src.exception import CustomException
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass 

#Step1: Create path variables 

@dataclass 
class DataIngestionconfig:
    raw_data_path:str = os.path.join('artifacts', 'raw.csv')
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self): 
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self): 
        logging.info("Data ingestion has started")
        try: 
            df = pd.read_csv(os.path.join('notebooks/data','gemstone.csv'))
            logging.info("Dataset has been read")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok= True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False)
            logging.info(" Train test split started")
            train_set,test_set =train_test_split(df, test_size= 0.25, random_state= 41)
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = False)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = False)

            logging.info("Ingestion of data has been completed")

            return(
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path
            ) 

        except Exception as e: 
            logging.info("Exception occured in the Data Ingestion Stage")
            raise CustomException(e, sys)