from src.exception.exception import NetworkSecurityException
from src.entity.artifact_entity import ClassificationMetricArtifact

import sys
from sklearn.metrics import f1_score, precision_score, recall_score

def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        model_f1_score = f1_score(y_true, y_pred, average='weighted')
        model_precision_score = precision_score(y_true, y_pred, average='weighted')
        model_recall_score = recall_score(y_true, y_pred, average='weighted')

        classification_metric = ClassificationMetricArtifact(f1_score=model_f1_score, precision_score=model_precision_score, recall_score=model_recall_score)

        return classification_metric
    
    except Exception as e:
        raise NetworkSecurityException(e, sys)