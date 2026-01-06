import sys
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_tranformation import DataTransformation
from src.components.model_trainer import Modeltrainer
from src.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig

if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiate the data Ingestion")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data Initiation Completed")
        print(data_ingestion_artifact, '\n')

        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        logging.info("Initiate the data Validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data Validation Completed")
        print(data_validation_artifact, '\n')

        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        logging.info("Initiate Data Transformation")
        data_transformation_artifact = data_transformation.initiate_data_tranformation()
        logging.info("Data Transformation Complete")
        print(data_transformation_artifact, '\n')

        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = Modeltrainer(model_trainer_config, data_transformation_artifact)
        logging.info("Initiate Model Training")
        model_training_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model Training Complete")
        print(model_training_artifact, '\n')



    except Exception as e:
        raise NetworkSecurityException(e, sys)
