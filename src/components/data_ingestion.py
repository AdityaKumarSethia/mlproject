import os
import sys

from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformer, DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','raw.csv')
    
    
class DataIngestion:
    def __init__(self):
            self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingest(self):
        logging.info("Entered the Data Ingestion method or component")
        
        try:
            df = pd.read_csv("notebook/dataset/students.csv")
            logging.info("Read the Dataset as dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("Train_Test Split initiated")
            train_set, test_set = train_test_split(df,test_size=0.2, random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index = False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False, header=True)
            
            logging.info("Ingestion Completed")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
            
        except Exception as e:
            raise CustomException(e,sys)
        
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingest()
    
    data_transform = DataTransformer()
    train_arr, test_arr, path = data_transform.initiate_data_transformation(train_data_path,test_data_path)
    
    model_trainer = ModelTrainer()
    score = model_trainer.initiate_model_trainer(train_arr,test_arr,path)
    
    print(score)
    