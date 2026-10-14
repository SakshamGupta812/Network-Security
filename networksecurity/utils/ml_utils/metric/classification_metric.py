
import sys

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score
)

from networksecurity.exception.exception import NetworkSecurityException


def get_classification_score(y_true, y_pred):
    try:
        # Explicitly identify the positive class.
        # In this project, phishing is the positive class.
        positive_label = "phishing"

        model_f1_score = f1_score(
            y_true,
            y_pred,
            pos_label=positive_label,
            zero_division=0
        )

        model_precision_score = precision_score(
            y_true,
            y_pred,
            pos_label=positive_label,
            zero_division=0
        )

        model_recall_score = recall_score(
            y_true,
            y_pred,
            pos_label=positive_label,
            zero_division=0
        )

        return {
            "f1_score": model_f1_score,
            "precision_score": model_precision_score,
            "recall_score": model_recall_score
        }

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e

