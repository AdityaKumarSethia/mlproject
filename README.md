# End To End Machine Learning Project structure

## Problem Statement
- This project understands how the students performance (test scores) is affected by other variables such as Gender, Ethinicity, Parental Level of Education, Lunch and Test Preparation course.


## Global Repo Structure

```
/repo
|
├── /src
|
├── .gitignore  # Make on Github using template and git pull to local repo
|
├── setup.py  # For distribution of project
|
├── requirments.txt  # List of packages for reproducibility
|
├── /notebook  # All `.ipynb` notebooks
|
└── README.md

```


#### Local Repo Structure (May not be visible in GitHub)

```
/repo
|
├── /src
├── .gitignore 
├── setup.py  
├── requirments.txt  
├── /notebook  
├── README.md
|
├── .venv  # Local Running Environment
|
├── generic-project.egg-info  # Distribution Created using setup.py
|
├── Logs  # Logging for a project
|
└── .vscode  # VS-CODE settings

```

### Inside `src` folder
```
/src
|
├── /components
|   ├── __init__.py
|   ├── data_ingestion.py
|   ├── data_transformation.py
|   └── model_trainer.py
|
├── /pipline
|   ├── __init__.py
|   ├── train_pipeline.py
|   └── test_pipeline.py
|  
├── logger.py
├── exception.py
└── utils.py
```

> `components` : all the modules and processes that we will create like data_ingestion etc. \
> `pipeline` : This directory orchestrates the components. While components defines what to do, pipeline defines in what order to do it. \
> `logger.py` : To set up a centralized logging system. \
> `exception.py` : To create a custom error handling system. \
> `utils.py` : A _toolbox_ for utility functions. 


### Inside `notebook` folder
```
/notebook
|
├── /dataset
|   └── student.csv  # dataset for the project
|
├── EDA_notebook.ipynb  # EDA & Data Cleaning
|
└── Model_Training.ipynb  # Model Implementation
```
