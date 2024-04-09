import os 
import sys 
from dataclasses import dataclass 
import pickle

import numpy as np 
import pandas as pd 
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from src.exception import CustomException 
from src.logger import logging 
from src.utils import save_function


@dataclass 
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation: 
    def __init__(self): 
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformation_object(self): 
        try: 
            logging.info("Data Transformation has been initiated") 
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']
            
            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            
            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler',StandardScaler())
                ]

            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            
            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e: 
            logging.info("Error occured in Data Transformation class")
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try: 
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Obtaining the preprocessing object")

            preprocessing_obj = self.get_data_transformation_object()
            target_colum_name = 'price'
            drop_columns = [target_colum_name, 'id']

            input_feature_train_df = train_df.drop(columns= drop_columns, axis = 1)
            target_colum_name_train_df = train_df[target_colum_name]

            input_feature_test_df = test_df.drop(columns= drop_columns, axis = 1)
            target_colum_name_test_df = test_df[target_colum_name]

            ## 
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying the preprocessing object to train and test datasets")

            train_arr = np.c_[input_feature_train_arr, np.array(target_colum_name_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_colum_name_test_df)]

            save_function(
                file_path= self.data_transformation_config.preprocessor_obj_file_path, obj = preprocessing_obj
            )   
    
        except Exception as e:
            logging.info("Error occured in initiate data transformation function")
            raise CustomException(e,sys)