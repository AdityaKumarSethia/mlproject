# System utils
import sys
import os
from dataclasses import dataclass
from typing import Tuple

# Data Wrangling libs
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Data transformation libs
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# SRC imports
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")
    
class DataTransformer:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_obj(self) -> ColumnTransformer:
        """
            This Function is responsible to return a Column Tranformer fitting the data
            
            Returns
            -------
                ColumnTransformer: One while will encode and standardize the values for the data.csv
        """
        try: 
            # Defining Features
            numerical_features = ["writing_score","reading_score"]
            categorical_features = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            
            # Pipelines
            num_pipeline = Pipeline(
                steps=[
                    ("num_imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            logging.info("Numeric columns scaling pipeline inititalized")
            
            cat_pipeline = Pipeline(
                steps=[
                    ("cat_imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(sparse=False)),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical columns encoding pipeline inititalized")
            
            # Combining Pipeline using Column Transformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline, numerical_features),
                    ("cat_pipeline",cat_pipeline, categorical_features)
                ]
            )
            
            logging.info("Pipelines Merged and returned")
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
        
        
    def initiate_data_transformation(self,train_path:str,test_path:str) -> Tuple[NDArray,NDArray,str]:
        """
        Return a transformed Data for model training and testing
        
        Args
        ----
            train_path : str
                file path to training data file
            test_path : str
                file path to testing data file
            
        Returns
        -------
            tuple[NDArray, NDArray, str]:
                NDArray -> Transformed Training Data Array
                
                NDArray -> Transformed Testing Data Array
                
                str -> file path to preprocessor object in .pkl
        """
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Reading Train and Test Data into a DataFrame")
            
            logging.info("Obtaining Preprocessor OBJ")
            
            preprocessor_obj = self.get_data_transformer_obj()
            
            target_column_name = "math_score"
            numerical_features = ["writing_score","reading_score"]
            
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info(f"Applying preprocessor obj on training and testing DataFrames")
            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            
            logging.info("Saved Preprocessor OBJ")
            
            save_obj(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)