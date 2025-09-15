import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_obj

class PredictPipeline:
    def __init__(self):
        ...
        
    def predict(self, features:pd.DataFrame) -> float:
        """
        Returns the prediction based on the Features
        
        Args
        ----
            features : DataFrame
                Features on which prediction must be performed
        
        Returns
        -------
            float
                target prediction (math's Score)
        """
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            model = load_obj(file_path=model_path)
            preprocessor = load_obj(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(
        self,
        gender : str,
        race_ethnicity : str,
        parental_level_of_education : str,
        lunch : str,
        test_preparation_course : str,
        reading_score : int,
        writing_score : int
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    def get_data_as_df(self) -> pd.DataFrame:
        """
        Returns a dataframe obj for the input pipeline
        """
        try:
            data_dict = {
                'gender' : [self.gender],
                'race_ethnicity' : [self.race_ethnicity],
                'parental_level_of_education' : [self.parental_level_of_education],
                'lunch' : [self.lunch],
                'test_preparation_course' : [self.test_preparation_course],
                'reading_score' : [self.reading_score],
                'writing_score' : [self.writing_score]
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e,sys)