from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact

from src.utils.main_utils.utils import load_numpy_array_data, evaluate_model, save_object, load_object
from src.utils.ml_utils.model_estimator import NetworkModel
from src.utils.ml_utils.classification_metric import get_classification_score


import os, sys
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature



class Modeltrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        

    def track_mlflow(self, best_model, best_model_name, train_metric, test_metric, model_params, X_train, y_train):
        """
        Track ML experiments with comprehensive metrics and model information.
        
        Args:
            best_model: Trained model object
            best_model_name: Name of the best model
            train_metric: Classification metrics for training data
            test_metric: Classification metrics for test data
            model_params: Dictionary of model hyperparameters
            X_train: Training features (for signature inference)
            y_train: Training labels (for signature inference)
        """
        with mlflow.start_run():
            # Log model name and tags
            mlflow.set_tag("model_name", best_model_name)
            mlflow.set_tag("model_type", "classification")
            mlflow.set_tag("framework", "sklearn")
            
            # Log model parameters
            mlflow.log_params(model_params)
            
            # Log training metrics
            mlflow.log_metric("train_f1_score", train_metric.f1_score)
            mlflow.log_metric("train_precision", train_metric.precision_score)
            mlflow.log_metric("train_recall", train_metric.recall_score)
            
            # Log test metrics
            mlflow.log_metric("test_f1_score", test_metric.f1_score)
            mlflow.log_metric("test_precision", test_metric.precision_score)
            mlflow.log_metric("test_recall", test_metric.recall_score)
            
            # Calculate and log overfitting metrics
            f1_diff = abs(train_metric.f1_score - test_metric.f1_score)
            mlflow.log_metric("f1_score_diff", f1_diff)
            mlflow.log_metric("overfitting_indicator", f1_diff > 0.1)
            
            # Infer and log model signature
            try:
                signature = infer_signature(X_train, best_model.predict(X_train))
                mlflow.sklearn.log_model(
                    best_model, 
                    "model",
                    signature=signature,
                    registered_model_name=f"NetworkSecurity_{best_model_name.replace(' ', '_')}"
                )
            except Exception as e:
                logging.info(f"Could not log model with signature: {e}")
                mlflow.sklearn.log_model(best_model, "model")
            
            logging.info(f"MLflow tracking completed for {best_model_name}")

    def train_model(self, X_train, y_train, X_test, y_test):
        # Configure MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("Network-Security-System")
        
        models = {
            "Random Forest": RandomForestClassifier(verbose = 1, random_state=42), 
            "Decision Tree": DecisionTreeClassifier(random_state=42), 
            "Gradient Boosting": GradientBoostingClassifier(verbose=1, random_state=42), 
            "Logistic Regression": LogisticRegression(verbose=1, max_iter=1000, random_state=42),
            "AdaBoost": AdaBoostClassifier(random_state=42)
        }

        # params = {
        #     "Decision Tree": {
        #         'criterion':['gini', 'entropy'],
        #         'max_depth': [3, 5, 7, 10],
        #         'min_samples_split': [2, 5, 10],
        #         'min_samples_leaf': [1, 2, 4],
        #         'max_features':['sqrt','log2']
        #     },
        #     "Random Forest":{
        #         'criterion':['gini', 'entropy'],
        #         'max_depth': [10, 20, 30],
        #         'min_samples_split': [2, 5, 10],
        #         'min_samples_leaf': [1, 2, 4],
        #         'max_features':['sqrt','log2'],
        #         'n_estimators': [50, 100, 200]
        #     },
        #     "Gradient Boosting":{
        #         'learning_rate':[0.01, 0.05, 0.1],
        #         'n_estimators': [50, 100, 200],
        #         'max_depth': [3, 5, 7],
        #         'min_samples_split': [2, 5],
        #         'min_samples_leaf': [1, 2],
        #         'subsample':[0.8, 0.9, 1.0],
        #         'max_features':['sqrt','log2']
        #     },
        #     "Logistic Regression":{
        #         'C': [0.01, 0.1, 1, 10, 100],
        #         'penalty': ['l2'],
        #         'solver': ['lbfgs', 'liblinear']
        #     },
        #     "AdaBoost":{
        #         'learning_rate':[0.01, 0.05, 0.1, 0.5, 1.0],
        #         'n_estimators': [50, 100, 200]
        #     }
        # }

        params={
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                # 'criterion':['gini', 'entropy', 'log_loss'],
                
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,128,256]
            },
            "Gradient Boosting":{
                # 'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.85,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [8,16,32,64,128,256]
            }
            
        }

        model_report, tuned_model = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)


        best_model_name = max(model_report, key=model_report.get)
        best_model = tuned_model[best_model_name]

        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
        
        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
        
        # Get model parameters
        model_params = best_model.get_params()
        
        ## Track the experiment with mlflow (single run for both train and test)
        self.track_mlflow(
            best_model=best_model,
            best_model_name=best_model_name,
            train_metric=classification_train_metric,
            test_metric=classification_test_metric,
            model_params=model_params,
            X_train=X_train,
            y_train=y_train
        )

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        network_model = NetworkModel(preprocessor=preprocessor, model=best_model)

        save_object(self.model_trainer_config.trained_model_file_path, obj = network_model)

        save_object("final_model/model.pkl", best_model)

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path= self.model_trainer_config.trained_model_file_path, 
            train_metric_artifact= classification_train_metric, 
            test_metric_artifact=classification_test_metric
        )

        logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
        return model_trainer_artifact


    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # loading Training and Testting array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test  = (
                train_arr[:, :-1], 
                train_arr[:, -1], 
                test_arr[:, :-1], 
                test_arr[:, -1]
            )

            model_trainer_artifact=self.train_model(X_train,y_train,X_test,y_test)
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)