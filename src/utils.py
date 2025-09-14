import os
import sys

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from typing import Callable, Dict
import dill
from sklearn.metrics import r2_score

from src.exception import CustomException

def save_obj(file_path:str, obj:object) -> None:
    """
    Saves a object in the given file path in form of a pickle file
    Args
    ----
        file_path : str
            desired file path where your want to save the object
        obj : object
            OBJECT which is going to be saved as pickle file
    """
    
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
    
def evaluate_models(X_train:NDArray, y_train:NDArray, X_test:NDArray, y_test:NDArray, models:Dict[str,BaseEstimator],metric:Callable=r2_score) -> Dict[str,float]:
    """Evaluates multiple models based on the given metrics
    
    Args
    ----
        X_train : array_like
            input Training data
        y_train : array_like
            input Target data
        X_test : array_like
            Testing data
        y_test : array_like
            Testing target data
        models : Dict[str, BaseEstimator]
            dictionary for all the models
        metrics : Callable, default=r2_score
            metric on which report will be generated on
            
    Returns
    -------
         model_name : str
             name for the model
         score : float
             score for the given metric
    """
    try:
        report = dict()
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train,y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_score = metric(y_train,y_train_pred)
            test_score = metric(y_test,y_test_pred)
            
            report[list(models.keys())[i]] = test_score
            
        return report
    except Exception as e:
        raise CustomException(e,sys)