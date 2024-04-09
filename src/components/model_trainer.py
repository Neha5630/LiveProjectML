import pandas as pd 
import numpy as np 
import sys 
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet 
from sklearn.tree import DecisionTreeRegressor
from src.logger import logging 
from src.exception import CustomException 
from dataclasses import dataclass
from src.utils import save_function 
from src.utils import model_performance 

@dataclass 
class ModelTrainerConfig():
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array): 
        try: 
            logging.info("Seggregating the dependent and independent variables")
            X_train, y_train, X_test, y_test = (train_array[:, :-1], 
                                                train_array[:,-1], 
                                                test_array[:, :-1], 
                                                test_array[:,-1])
            models = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(), 
                "Lasso":Lasso(), 
                "ElasticNet": ElasticNet(), 
                "DecisionTree": DecisionTreeRegressor()
            }
            model_report: dict = model_performance(X_train, y_train, X_test, y_test, models)

            print(model_report)
            print("\n"*100)
            logging.info(f"Model Report: {model_report}")

            # Best model
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)] # Study again
            
            best_model = models[best_model_name]

            print(f"The best model is {best_model_name}, with R2 Score: {best_model_score}")
            print("\n"*100)
            logging.info(f"The best model is {best_model_name}, with R2 Score: {best_model_score}")
            save_function(file_path= self.model_trainer_config.trained_model_file_path, obj = best_model)


        except Exception as e: 
            logging.info("Error occured during model training")
            raise CustomException(e,sys)