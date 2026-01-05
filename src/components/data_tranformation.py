from src.logging.logger import logging
from src.exception.exception import NetworkSecurityException

from src.constants.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact

from src.utils.main_utils.utils import save_numpy_array_data, save_object

import os, sys
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_tranformation_config = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        

    @staticmethod
    def read_data(file_path)-> pd.DataFrame :
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    @classmethod
    def get_data_transformer_object(cls) -> Pipeline :
        """
        It initialises a KNNImputer object with the parameters specifie in the training_pipeline.py file
        and return a pepeline object with the KNNImpiter object as the first step.

        Args: 
            cls: DataTransformation

        Returns:
            A Pipeline object
        """

        logging.info("Entered get_data_transformer_object method of tranformation class")

        try:
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            
            logging.info(f"Initialize KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}")

            processor = Pipeline([
                ("imputer", imputer)
            ])
            return processor
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    
    def initiate_data_tranformation(self) -> DataTransformationArtifact:
        
        logging.info("Entered initiate_data_transformation method of Data Transformation Class")

        try:
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # Training DataFrame
            input_feature_train_df = train_df.drop(TARGET_COLUMN, axis = 1)
            target_feature_train_df = train_df[TARGET_COLUMN].replace(-1, 0)

            # Testing DataFrame
            input_feature_test_df = test_df.drop(TARGET_COLUMN, axis = 1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            preprocessor = DataTransformation.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            # save numpy array data
            save_numpy_array_data(self.data_tranformation_config.transformed_train_file_path, array = train_arr)
            save_numpy_array_data(self.data_tranformation_config.transformed_test_file_path, array = test_arr)
            save_object(self.data_tranformation_config.transformed_object_file_path, preprocessor_object)

            save_object("final_model/preprocessor.pkl", preprocessor_object)

            # Preparing Artifacts   
            data_tranformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_tranformation_config.transformed_object_file_path, 
                transformed_train_file_path=self.data_tranformation_config.transformed_train_file_path, 
                transformed_test_file_path=self.data_tranformation_config.transformed_test_file_path
            )

            return data_tranformation_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)


    