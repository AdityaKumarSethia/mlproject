import os
import sys

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from typing import Any, Callable, Dict
import dill
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
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
    
    
def evaluate_models(X_train:NDArray, y_train:NDArray, X_test:NDArray, y_test:NDArray, models:Dict[str,BaseEstimator],params,metric:Callable=r2_score) -> Dict[str,float]:
    """
    Evaluates multiple models based on the given metrics. While preforming Hyperparamter tuning and generating a report with best model_param and the score
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
        params : Dict[str, Dict[str,list[Any]]]
            A nested Dict containing model name and the parameter on which Hyperparamter tuning will happen on
        metrics : Callable, default=r2_score
            metric on which report will be generated on
            
    Returns
    -------
        model_name : str
             name for the model
        score : float
             score for the given metric
        params : Dict
            Set of best params
    """
    try:
        report = dict()
        
        logging.info("Started Hyperparameter tuning for each model")
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param=params[list(models.keys())[i]]
            name = list(models.keys())[i]
            
            logging.info(f"Hyperparmater tuning for {name} commences")
            gs = GridSearchCV(model,param,cv=3)
            gs.fit(X_train,y_train)

            logging.info(f"Hyperparmeter Tuning for {name} ends")
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_score = metric(y_train,y_train_pred)
            test_score = metric(y_test,y_test_pred)
            
            report[list(models.keys())[i]] = (test_score,gs.best_params_)
            
        return report
    except Exception as e:
        raise CustomException(e,sys)