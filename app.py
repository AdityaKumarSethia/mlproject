from flask import Flask, request, render_template
from graphviz import render
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_data',methods=['GET','POST'])
def predict_data():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Input validation: check for missing values
        required_fields = [
            'gender', 'race_ethnicity', 'parental_level_of_education',
            'lunch', 'test_preparation_course', 'reading_score', 'writing_score'
        ]
        missing = [field for field in required_fields if not request.form.get(field)]
        if missing:
            return render_template('home.html', results=None, error=f"Missing fields: {', '.join(missing)}")

        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        pred_df = data.get_data_as_df()
        print(pred_df)
        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=result[0], error=None)
    
    
if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
