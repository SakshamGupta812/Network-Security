import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os,sys
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def read_yaml_file(file_path : str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e


def write_yaml_file(file_path: str, content: object, replace:bool = False)-> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e,sys)

def save_numpy_array_data(file_path:str, array:np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e

def save_object(file_path:str, obj:object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Excited the save_object method of MinUtils class")
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e

def load_object(file_path : str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path,"rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e


def load_numpy_array_data(file_path : str) -> np.array:
    """
    Load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """

    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj, allow_pickle=True)
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import accuracy_score

        report = {}

        for model_name, model in models.items():

            # Get the parameter grid for the current model
            param_grid = param.get(model_name, {})

            # Correct order:
            # First argument = estimator
            # Second argument = parameter grid
            gs = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=3,
                scoring="accuracy",
                n_jobs=-1,
                error_score="raise"
            )

            # Train and tune the model
            gs.fit(X_train, y_train)

            # Retrieve the best tuned model
            best_model = gs.best_estimator_

            # Evaluate on the test set
            y_test_pred = best_model.predict(X_test)
            test_model_score = accuracy_score(y_test, y_test_pred)

            # Save the tuned model, not the original model
            models[model_name] = best_model
            report[model_name] = test_model_score

            logging.info(
                f"Model: {model_name}, "
                f"Best Parameters: {gs.best_params_}, "
                f"Accuracy: {test_model_score}"
            )

        return report

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e

