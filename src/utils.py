import os
import sys

import numpy as np
import pandas as pd

import dill

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