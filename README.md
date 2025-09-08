# End To End MAchine Learning Project structure




### Inside `src` folder
```
/src
|
├── /components
|   ├── __init__.py
|   ├── data_ingestion.py
|   ├── data_transformation.py
|   ├── model_trainer.py
|
├── /pipline
|   ├── __init__.py
|   ├── train_pipeline.py
|   ├── test_pipeline.py
|  
├── logger.py
├── exception.py
├── utils.py
```

> `components` : all the modules and processes that we will create like data_ingestion etc. \
> `pipeline` : This directory orchestrates the components. While components defines what to do, pipeline defines in what order to do it. \
> `logger.py` : To set up a centralized logging system. \
> `exception.py` : To create a custom error handling system. \
> `utils.py` : A _toolbox_ for utility functions. 
