import os
import sys
from dataclasses import dataclass
from numpy.typing import NDArray


from catboost import CatBoostRegressor
from sklearn import preprocessing
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
    
    
class ModelTrainer:
    """
    Model Training Class
    """
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array:NDArray,test_array:NDArray,preprocessor_path:str) -> float:
        """
        Initiates Model Training and returns the r2_score for the best model
        
        Args
        ----
            train_array : array_like
                training dataset
            test_array : array_like
                testing dataset
            preprocessor_path : str
                path to preprocessor_obj if needed
        
        Returns
        -------
            float
                R2 Score of the best model
        """
        try:
            logging.info("Train Test Split")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "Linear Regression": LinearRegression(n_jobs=-1),
                "K-Neighbors Regressor": KNeighborsRegressor(n_jobs=-1),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(n_jobs=-1),
                "XGBRegressor": XGBRegressor(tree_method="hist",device='cuda'), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False,task_type="GPU"),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            
            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best Model found")
            
            logging.info(f"Best found model on both training and test dataset")
            
            save_obj(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            
            logging.info("Best Model pickled")
            
            predicted = best_model.predict(X_test)
            r2_result = r2_score(y_test,predicted)
            
            return r2_result
        
        except Exception as e:
            raise CustomException(e,sys)